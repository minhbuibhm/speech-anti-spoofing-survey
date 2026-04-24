from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path("sdd-refactor.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


cells: list[dict] = [
    md(
        """
        # Speech Deepfake Detection — Refactored Pipeline

        This notebook is organized around **artifacts first** and **datasets first**.

        Design goals:
        - Keep the code simple, readable, and easy to modify.
        - Separate expensive stages so they can be resumed from saved files.
        - Check mounted Kaggle artifacts before recomputing anything expensive.
        - Evaluate models dataset by dataset in this order: **ASVspoof 2019 → ASVspoof 2021 DF → ASVspoof 5**.
        - Run error analysis immediately after each dataset finishes.

        The notebook uses four models only:
        - `LFCC + LCNN`
        - `AASIST`
        - `AASIST-L`
        - `AASIST3`
        """
    ),
    md(
        """
        ## Part 0 — Setup

        This section defines the global configuration, artifact paths, and reusable helper functions.
        """
    ),
    code(
        '''
        import gc
        import glob
        import json
        import os
        import pickle
        import sys
        import time
        import importlib
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchaudio

        from sklearn.metrics import roc_curve
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        try:
            from datasets import load_dataset
        except ImportError:
            %pip install -q datasets soundfile transformers huggingface_hub
            from datasets import load_dataset

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        RUN_CONFIG = {
            "force_reindex": False,
            "force_train_lfcc": False,
            "force_eval": {
                "asv19": False,
                "asv21_df": False,
                "asv5": False,
            },
            "force_error_analysis": {
                "asv19": False,
                "asv21_df": False,
                "asv5": False,
            },
            "enable_models": ["AASIST", "AASIST-L", "AASIST3", "LFCC+LCNN"],
            "enable_asv5_hf_fallback": True,
            "asv5_split": "dev",
            "asv5_track": "track_1",
            "asv5_max_items": None,
            "lfcc_epochs": 10,
            "lfcc_batch_size": 64,
        }

        RESULTS_ROOT = Path("/kaggle/working/sdd_refactor_results")
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

        ARTIFACT_MOUNT_ROOTS = [
            Path("/kaggle/input"),
            Path("/kaggle/input/datasets/minhbhm"),
            Path("/kaggle/datasets/minhbhm"),
        ]

        # Each group can map to multiple Kaggle dataset slugs because the same
        # artifact may be mounted with underscores or hyphens depending on how it
        # was uploaded or attached to a notebook.
        ARTIFACT_DATASET_NAMES = {
            "asv19": ["asvspoof19_results", "asvspoof19-results", "sdd-refactor-asv19"],
            "asv21_df": ["asvspoof2021_results", "asvspoof2021-results", "asvspoof21-results", "sdd-refactor-asv21"],
            "asv5": ["asvspoof5-results", "sdd-refactor-asv5"],
            "checkpoints": ["sdd-checkpoints", "sdd-refactor-checkpoints", "asvspoof19_results", "asvspoof19-results"],
            "indexes": ["sdd-refactor-indexes"],
        }

        LEGACY_ASV19_PICKLES = [
            "all_results_asvspooft_19.pkl",
            "all_results.pkl",
        ]

        DATASET_OUTPUT_DIRS = {
            "asv19": RESULTS_ROOT / "asv19",
            "asv21_df": RESULTS_ROOT / "asv21_df",
            "asv5": RESULTS_ROOT / "asv5",
            "checkpoints": RESULTS_ROOT / "checkpoints",
            "indexes": RESULTS_ROOT / "indexes",
        }

        for path in DATASET_OUTPUT_DIRS.values():
            path.mkdir(parents=True, exist_ok=True)

        COMMON_METADATA_COLUMNS = [
            "dataset",
            "split",
            "speaker",
            "utt_id",
            "source",
            "attack",
            "codec",
            "vocoder",
            "label",
            "audio_path",
            "hf_index",
        ]
        '''
    ),
    code(
        '''
        def safe_name(value: str) -> str:
            """Convert a display name into a filesystem-safe token."""
            return value.replace("+", "_").replace(" ", "_").replace("/", "_")


        def ensure_parent(path: Path) -> Path:
            """Create the parent directory of a file path."""
            path.parent.mkdir(parents=True, exist_ok=True)
            return path


        def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
            """Compute Equal Error Rate in percent."""
            scores = np.asarray(scores)
            labels = np.asarray(labels)
            fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fpr - fnr))
            return float((fpr[idx] + fnr[idx]) / 2 * 100)


        def get_artifact_roots(group: str) -> list[Path]:
            """Return all artifact roots for a logical artifact group."""
            roots: list[Path] = []
            for root in ARTIFACT_MOUNT_ROOTS:
                for dataset_name in ARTIFACT_DATASET_NAMES.get(group, []):
                    roots.append(root / dataset_name)
            roots.append(DATASET_OUTPUT_DIRS[group])
            return roots


        def find_existing_file(group: str, relative_paths: list[str] | tuple[str, ...]) -> Path | None:
            """Search mounted datasets and local results for an existing artifact."""
            for root in get_artifact_roots(group):
                for relative_path in relative_paths:
                    candidate = root / relative_path
                    if candidate.exists():
                        return candidate
            return None


        def artifact_exists(group: str, relative_paths: list[str] | tuple[str, ...]) -> bool:
            """Check whether any candidate artifact exists."""
            return find_existing_file(group, relative_paths) is not None


        def save_json(path: Path, data: dict) -> None:
            """Save a JSON file with stable formatting."""
            ensure_parent(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


        def load_json(path: Path) -> dict:
            """Load a JSON file."""
            return json.loads(Path(path).read_text(encoding="utf-8"))


        def save_text(path: Path, text: str) -> None:
            """Save plain text."""
            ensure_parent(path).write_text(text, encoding="utf-8")


        def print_header(title: str) -> None:
            """Print a visible section header."""
            print("\\n" + "=" * 80)
            print(title)
            print("=" * 80)


        def release_model(model: nn.Module | None) -> None:
            """Release model memory as early as possible."""
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        def standardize_metadata(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
            """Ensure every dataset index has the same columns for evaluation and analysis."""
            df = df.copy()
            defaults = {
                "dataset": dataset_key,
                "split": "eval",
                "speaker": "unknown",
                "utt_id": "",
                "source": "unknown",
                "attack": "unknown",
                "codec": "nocodec",
                "vocoder": "unknown",
                "label": 0,
                "audio_path": None,
                "hf_index": np.nan,
            }
            for column, default in defaults.items():
                if column not in df.columns:
                    df[column] = default
            df["utt_id"] = df["utt_id"].astype(str)
            df["label"] = df["label"].astype(int)
            return df[COMMON_METADATA_COLUMNS].reset_index(drop=True)
        '''
    ),
    md(
        """
        ## Part 1 — Dataset Indexing

        This section loads dataset metadata only. Audio is **not** loaded into RAM up front.
        """
    ),
    code(
        '''
        def locate_asvspoof2019() -> dict:
            """Locate ASVspoof 2019 LA train and eval paths in Kaggle inputs."""
            candidates = [
                Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"),
                Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA"),
            ]
            for base in candidates:
                if not base.exists():
                    continue
                resolved_base = base / "LA" if (base / "LA").exists() and (base / "ASVspoof2019_LA_eval").exists() is False else base
                train_flac = resolved_base / "ASVspoof2019_LA_train" / "flac"
                eval_flac = resolved_base / "ASVspoof2019_LA_eval" / "flac"
                proto_dir = resolved_base / "ASVspoof2019_LA_cm_protocols"
                train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
                eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
                if train_flac.exists() and eval_flac.exists() and train_proto.exists() and eval_proto.exists():
                    return {
                        "base": resolved_base,
                        "train_flac": train_flac,
                        "eval_flac": eval_flac,
                        "train_proto": train_proto,
                        "eval_proto": eval_proto,
                    }
            raise FileNotFoundError("Could not locate ASVspoof 2019 LA in Kaggle inputs.")


        def locate_asvspoof2021_df() -> dict:
            """Locate ASVspoof 2021 DF evaluation audio and protocol files."""
            base = Path("/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021")
            if not base.exists():
                raise FileNotFoundError("Could not locate ASVspoof 2021 dataset root.")

            part_dirs: list[Path] = []
            for part_name in [
                "ASVspoof2021_DF_eval_part00",
                "ASVspoof2021_DF_eval_part01",
                "ASVspoof2021_DF_eval_part02",
            ]:
                part_root = base / part_name
                if not part_root.exists():
                    continue
                for dirpath, _, files in os.walk(part_root):
                    if any(file.endswith(".flac") for file in files):
                        part_dirs.append(Path(dirpath))
                        break

            trial_path = None
            keys_root = base / "DF-keys-full"
            if keys_root.exists():
                for dirpath, _, files in os.walk(keys_root):
                    for filename in files:
                        if "trial" in filename.lower() and filename.endswith(".txt"):
                            trial_path = Path(dirpath) / filename
                            break
                    if trial_path is not None:
                        break

            if not part_dirs or trial_path is None:
                raise FileNotFoundError("Could not locate ASVspoof 2021 DF audio or trial files.")

            return {
                "base": base,
                "audio_dirs": part_dirs,
                "trial": trial_path,
            }


        def locate_asvspoof5_local() -> dict | None:
            """Try to locate a mounted local copy of ASVspoof 5."""
            protocol_candidates = [
                f"/kaggle/input/**/*{RUN_CONFIG['asv5_split']}*{RUN_CONFIG['asv5_track']}*.tsv",
                f"/kaggle/input/**/*{RUN_CONFIG['asv5_track']}*{RUN_CONFIG['asv5_split']}*.tsv",
                f"/kaggle/input/**/*{RUN_CONFIG['asv5_split']}*.tsv",
            ]
            audio_candidates = [
                f"/kaggle/input/**/*flac*{RUN_CONFIG['asv5_split']}*",
                f"/kaggle/input/**/*{RUN_CONFIG['asv5_split']}*flac*",
            ]

            protocol_path = None
            audio_dir = None

            for pattern in protocol_candidates:
                matches = sorted(glob.glob(pattern, recursive=True))
                if matches:
                    protocol_path = Path(matches[0])
                    break

            for pattern in audio_candidates:
                matches = [Path(path) for path in sorted(glob.glob(pattern, recursive=True)) if Path(path).is_dir()]
                if matches:
                    audio_dir = matches[0]
                    break

            if protocol_path is None or audio_dir is None:
                return None

            return {
                "protocol": protocol_path,
                "audio_dir": audio_dir,
            }
        '''
    ),
    code(
        '''
        def parse_protocol_2019(protocol_path: Path, audio_dir: Path, split_name: str) -> pd.DataFrame:
            """Parse ASVspoof 2019 protocol and attach audio paths."""
            rows = []
            with open(protocol_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    utt_id = parts[1]
                    rows.append(
                        {
                            "dataset": "asv19",
                            "split": split_name,
                            "speaker": parts[0],
                            "utt_id": utt_id,
                            "source": "asvspoof2019",
                            "attack": parts[3] if parts[3] != "-" else "bonafide",
                            "codec": "nocodec",
                            "vocoder": "unknown",
                            "label": 1 if parts[4] == "bonafide" else 0,
                            "audio_path": str(audio_dir / f"{utt_id}.flac"),
                            "hf_index": np.nan,
                        }
                    )
            return standardize_metadata(pd.DataFrame(rows), "asv19")


        def parse_protocol_2021_df(trial_path: Path, audio_dirs: list[Path]) -> pd.DataFrame:
            """Parse ASVspoof 2021 DF protocol and filter to files that really exist."""
            audio_index: dict[str, str] = {}
            for audio_dir in audio_dirs:
                for filename in os.listdir(audio_dir):
                    if filename.endswith(".flac"):
                        audio_index[filename[:-5]] = str(audio_dir / filename)

            rows = []
            with open(trial_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    label_str = parts[5]
                    if label_str not in ("bonafide", "spoof"):
                        continue
                    utt_id = parts[1]
                    if utt_id not in audio_index:
                        continue
                    rows.append(
                        {
                            "dataset": "asv21_df",
                            "split": "eval",
                            "speaker": parts[0],
                            "utt_id": utt_id,
                            "source": parts[3] if len(parts) > 3 else "unknown",
                            "attack": parts[4] if len(parts) > 4 else "unknown",
                            "codec": parts[2] if len(parts) > 2 else "unknown",
                            "vocoder": parts[8] if len(parts) > 8 else "unknown",
                            "label": 1 if label_str == "bonafide" else 0,
                            "audio_path": audio_index[utt_id],
                            "hf_index": np.nan,
                        }
                    )
            return standardize_metadata(pd.DataFrame(rows), "asv21_df")


        def normalize_asv5_label(value) -> int:
            """Normalize an ASVspoof 5 label into the bonafide/spoof binary convention."""
            return 1 if "bona" in str(value).strip().lower() else 0


        def normalize_asv5_group(value, default: str = "unknown") -> str:
            """Normalize missing or empty metadata values."""
            text = str(value).strip()
            return default if text in ("", "-", "None", "nan") else text


        def find_first_key(sample: dict, keys: list[str], fallback: str | None = None) -> str | None:
            """Find the first matching key in a Hugging Face sample."""
            for key in keys:
                if key in sample:
                    return key
            return fallback


        def index_asvspoof5_local(protocol_path: Path, audio_dir: Path) -> pd.DataFrame:
            """Build an ASVspoof 5 metadata table from mounted local files."""
            df = pd.read_csv(protocol_path, sep="\\t")
            if "utt_id" not in df.columns or "label" not in df.columns:
                df = pd.read_csv(protocol_path, sep="\\t", header=None).rename(
                    columns={
                        0: "speaker",
                        1: "utt_id",
                        5: "codec",
                        6: "attack",
                        7: "label",
                    }
                )

            audio_index = {path.stem: str(path) for path in Path(audio_dir).rglob("*.flac")}
            df["audio_path"] = df["utt_id"].map(audio_index)
            df = df[df["audio_path"].notna()].copy()
            if RUN_CONFIG["asv5_max_items"] is not None:
                df = df.iloc[: RUN_CONFIG["asv5_max_items"]].copy()

            df = df.assign(
                dataset="asv5",
                split=RUN_CONFIG["asv5_split"],
                speaker=df.get("speaker", "unknown"),
                source=df.get("source", "local_files"),
                attack=df.get("attack", "unknown").map(lambda value: normalize_asv5_group(value, "bonafide")),
                codec=df.get("codec", "unknown").map(lambda value: normalize_asv5_group(value, "nocodec")),
                vocoder="unknown",
                label=df["label"].map(normalize_asv5_label),
                hf_index=np.nan,
            )
            return standardize_metadata(df, "asv5")


        def load_asvspoof5_hf_handle():
            """Load the requested ASVspoof 5 split from Hugging Face."""
            split_candidates = [
                RUN_CONFIG["asv5_split"],
                f"{RUN_CONFIG['asv5_split']}.{RUN_CONFIG['asv5_track']}",
                f"{RUN_CONFIG['asv5_track']}_{RUN_CONFIG['asv5_split']}",
            ]
            hf_dataset = None
            last_error = None
            for split_name in split_candidates:
                try:
                    hf_dataset = load_dataset("jungjee/asvspoof5", split=split_name)
                    print(f"Loaded ASVspoof 5 from Hugging Face split: {split_name}")
                    break
                except Exception as exc:
                    last_error = exc
            if hf_dataset is None:
                raise RuntimeError(f"Cannot load ASVspoof 5 from Hugging Face: {last_error}")
            return hf_dataset


        def index_asvspoof5_hf() -> tuple[pd.DataFrame, object]:
            """Build an ASVspoof 5 metadata table from Hugging Face."""
            hf_dataset = load_asvspoof5_hf_handle()

            rows = []
            iterator = hf_dataset
            if RUN_CONFIG["asv5_max_items"] is not None:
                iterator = hf_dataset.select(range(min(RUN_CONFIG["asv5_max_items"], len(hf_dataset))))

            for idx, sample in enumerate(iterator):
                utt_key = find_first_key(sample, ["utt_id", "id", "file", "filename"], "utt_id")
                attack_key = find_first_key(sample, ["attack", "attack_id", "spoofing_attack"], None)
                codec_key = find_first_key(sample, ["codec", "codec_id", "compression"], None)
                label_key = find_first_key(sample, ["label", "bonafide", "class", "target"], "label")
                rows.append(
                    {
                        "dataset": "asv5",
                        "split": RUN_CONFIG["asv5_split"],
                        "speaker": str(sample.get("speaker", "unknown")),
                        "utt_id": str(sample.get(utt_key, f"sample_{idx}")),
                        "source": normalize_asv5_group(sample.get("source", "huggingface"), "huggingface"),
                        "attack": normalize_asv5_group(sample.get(attack_key, "unknown"), "unknown"),
                        "codec": normalize_asv5_group(sample.get(codec_key, "nocodec"), "nocodec"),
                        "vocoder": normalize_asv5_group(sample.get("vocoder", "unknown"), "unknown"),
                        "label": normalize_asv5_label(sample.get(label_key, 0)),
                        "audio_path": None,
                        "hf_index": idx,
                    }
                )
            return standardize_metadata(pd.DataFrame(rows), "asv5"), hf_dataset


        def save_index(df: pd.DataFrame, relative_path: str) -> Path:
            """Save a dataset index into the local results directory."""
            path = DATASET_OUTPUT_DIRS["indexes"] / relative_path
            ensure_parent(path)
            df.to_csv(path, index=False)
            return path


        def load_or_build_dataset_indexes() -> dict:
            """Load cached indexes if they exist, otherwise rebuild them."""
            bundles: dict[str, dict] = {}

            # ASVspoof 2019
            index_19_train = find_existing_file("indexes", ["asv19_train_index.csv"])
            index_19_eval = find_existing_file("indexes", ["asv19_eval_index.csv"])
            if index_19_train is not None and index_19_eval is not None and not RUN_CONFIG["force_reindex"]:
                train_df = pd.read_csv(index_19_train)
                eval_df = pd.read_csv(index_19_eval)
                loc_19 = locate_asvspoof2019()
                bundles["asv19"] = {
                    "name": "ASVspoof 2019 LA",
                    "train_df": train_df,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack"],
                    "source": "cached_index",
                    "paths": loc_19,
                    "hf_dataset": None,
                }
            else:
                loc_19 = locate_asvspoof2019()
                train_df = parse_protocol_2019(loc_19["train_proto"], loc_19["train_flac"], "train")
                eval_df = parse_protocol_2019(loc_19["eval_proto"], loc_19["eval_flac"], "eval")
                save_index(train_df, "asv19_train_index.csv")
                save_index(eval_df, "asv19_eval_index.csv")
                bundles["asv19"] = {
                    "name": "ASVspoof 2019 LA",
                    "train_df": train_df,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack"],
                    "source": "rebuilt_index",
                    "paths": loc_19,
                    "hf_dataset": None,
                }

            # ASVspoof 2021 DF
            index_21 = find_existing_file("indexes", ["asv21_df_eval_index.csv"])
            if index_21 is not None and not RUN_CONFIG["force_reindex"]:
                eval_df = pd.read_csv(index_21)
                loc_21 = locate_asvspoof2021_df()
                bundles["asv21_df"] = {
                    "name": "ASVspoof 2021 DF",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec", "vocoder"],
                    "source": "cached_index",
                    "paths": loc_21,
                    "hf_dataset": None,
                }
            else:
                loc_21 = locate_asvspoof2021_df()
                eval_df = parse_protocol_2021_df(loc_21["trial"], loc_21["audio_dirs"])
                save_index(eval_df, "asv21_df_eval_index.csv")
                bundles["asv21_df"] = {
                    "name": "ASVspoof 2021 DF",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec", "vocoder"],
                    "source": "rebuilt_index",
                    "paths": loc_21,
                    "hf_dataset": None,
                }

            # ASVspoof 5
            index_5 = find_existing_file("indexes", [f"asv5_{RUN_CONFIG['asv5_split']}_index.csv"])
            local_asv5 = locate_asvspoof5_local()
            if index_5 is not None and not RUN_CONFIG["force_reindex"]:
                eval_df = pd.read_csv(index_5)
                hf_dataset = None
                if "audio_path" not in eval_df.columns:
                    eval_df["audio_path"] = None
                if "hf_index" not in eval_df.columns:
                    eval_df["hf_index"] = np.nan
                if eval_df["audio_path"].isna().all() and local_asv5 is None and RUN_CONFIG["enable_asv5_hf_fallback"]:
                    hf_dataset = load_asvspoof5_hf_handle()
                bundles["asv5"] = {
                    "name": f"ASVspoof 5 ({RUN_CONFIG['asv5_split']})",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec"],
                    "source": "cached_index",
                    "paths": local_asv5,
                    "hf_dataset": hf_dataset,
                }
            else:
                hf_dataset = None
                if local_asv5 is not None:
                    eval_df = index_asvspoof5_local(local_asv5["protocol"], local_asv5["audio_dir"])
                    source = "local_files"
                else:
                    if not RUN_CONFIG["enable_asv5_hf_fallback"]:
                        raise FileNotFoundError("ASVspoof 5 local files were not found and HF fallback is disabled.")
                    eval_df, hf_dataset = index_asvspoof5_hf()
                    source = "huggingface"
                save_index(eval_df, f"asv5_{RUN_CONFIG['asv5_split']}_index.csv")
                bundles["asv5"] = {
                    "name": f"ASVspoof 5 ({RUN_CONFIG['asv5_split']})",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec"],
                    "source": source,
                    "paths": local_asv5,
                    "hf_dataset": hf_dataset,
                }

            return bundles
        '''
    ),
    code(
        '''
        DATASET_BUNDLE_CACHE: dict[str, dict] = {}


        def _asv5_cached_mode(eval_df: pd.DataFrame) -> str:
            """Infer whether a cached ASVspoof 5 index points to local files or Hugging Face rows."""
            if "audio_path" not in eval_df.columns or eval_df["audio_path"].isna().all():
                return "huggingface"
            existing_paths = eval_df["audio_path"].dropna().astype(str).head(20)
            if any(Path(path).exists() for path in existing_paths):
                return "local_files"
            return "stale_local_files"


        def build_dataset_bundle(dataset_key: str) -> dict:
            """Build or load one dataset bundle without touching the other datasets."""
            if dataset_key == "asv19":
                index_train = find_existing_file("indexes", ["asv19_train_index.csv"])
                index_eval = find_existing_file("indexes", ["asv19_eval_index.csv"])
                loc = locate_asvspoof2019()
                if index_train is not None and index_eval is not None and not RUN_CONFIG["force_reindex"]:
                    train_df = standardize_metadata(pd.read_csv(index_train), "asv19")
                    eval_df = standardize_metadata(pd.read_csv(index_eval), "asv19")
                    source = "cached_index"
                else:
                    train_df = parse_protocol_2019(loc["train_proto"], loc["train_flac"], "train")
                    eval_df = parse_protocol_2019(loc["eval_proto"], loc["eval_flac"], "eval")
                    save_index(train_df, "asv19_train_index.csv")
                    save_index(eval_df, "asv19_eval_index.csv")
                    source = "rebuilt_index"
                return {
                    "name": "ASVspoof 2019 LA",
                    "train_df": train_df,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack"],
                    "source": source,
                    "paths": loc,
                    "hf_dataset": None,
                }

            if dataset_key == "asv21_df":
                index_eval = find_existing_file("indexes", ["asv21_df_eval_index.csv"])
                loc = locate_asvspoof2021_df()
                if index_eval is not None and not RUN_CONFIG["force_reindex"]:
                    eval_df = standardize_metadata(pd.read_csv(index_eval), "asv21_df")
                    source = "cached_index"
                else:
                    eval_df = parse_protocol_2021_df(loc["trial"], loc["audio_dirs"])
                    save_index(eval_df, "asv21_df_eval_index.csv")
                    source = "rebuilt_index"
                return {
                    "name": "ASVspoof 2021 DF",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec", "vocoder"],
                    "source": source,
                    "paths": loc,
                    "hf_dataset": None,
                }

            if dataset_key == "asv5":
                index_eval = find_existing_file("indexes", [f"asv5_{RUN_CONFIG['asv5_split']}_index.csv"])
                local_asv5 = locate_asvspoof5_local()
                hf_dataset = None
                rebuild = RUN_CONFIG["force_reindex"] or index_eval is None
                eval_df = None
                source = "unknown"

                if not rebuild:
                    eval_df = standardize_metadata(pd.read_csv(index_eval), "asv5")
                    cached_mode = _asv5_cached_mode(eval_df)
                    if cached_mode == "huggingface":
                        if not RUN_CONFIG["enable_asv5_hf_fallback"]:
                            rebuild = True
                        else:
                            hf_dataset = load_asvspoof5_hf_handle()
                            source = "cached_index_huggingface"
                    elif cached_mode == "local_files":
                        source = "cached_index_local_files"
                    else:
                        rebuild = True

                if rebuild:
                    if local_asv5 is not None:
                        eval_df = index_asvspoof5_local(local_asv5["protocol"], local_asv5["audio_dir"])
                        source = "local_files"
                    else:
                        if not RUN_CONFIG["enable_asv5_hf_fallback"]:
                            raise FileNotFoundError("ASVspoof 5 local files were not found and HF fallback is disabled.")
                        eval_df, hf_dataset = index_asvspoof5_hf()
                        source = "huggingface"
                    save_index(eval_df, f"asv5_{RUN_CONFIG['asv5_split']}_index.csv")

                return {
                    "name": f"ASVspoof 5 ({RUN_CONFIG['asv5_split']})",
                    "train_df": None,
                    "eval_df": eval_df,
                    "analysis_groups": ["attack", "codec", "vocoder"],
                    "source": source,
                    "paths": local_asv5,
                    "hf_dataset": hf_dataset,
                }

            raise KeyError(f"Unknown dataset key: {dataset_key}")


        def get_dataset_bundle(dataset_key: str) -> dict:
            """Return one dataset bundle, building it lazily if needed."""
            if dataset_key not in DATASET_BUNDLE_CACHE:
                DATASET_BUNDLE_CACHE[dataset_key] = build_dataset_bundle(dataset_key)
                describe_dataset_bundle(dataset_key, DATASET_BUNDLE_CACHE[dataset_key])
            return DATASET_BUNDLE_CACHE[dataset_key]


        def describe_dataset_bundle(dataset_key: str, bundle: dict) -> None:
            """Print a compact dataset index summary."""
            eval_df = bundle["eval_df"]
            train_rows = len(bundle["train_df"]) if bundle["train_df"] is not None else 0
            print(f"{dataset_key:10s} | source={bundle['source']:12s} | train={train_rows:,} | eval={len(eval_df):,}")
            for column in ["attack", "codec", "vocoder"]:
                if column in eval_df.columns:
                    n_unique = eval_df[column].nunique(dropna=False)
                    print(f"  - {column}: {n_unique} groups")
        '''
    ),
    md(
        """
        ## Part 2 — Model Loading

        This section defines the four model loaders. LFCC + LCNN is loaded from a checkpoint if available; otherwise it will be trained in the next section.
        """
    ),
    code(
        '''
        class SimpleLCNN(nn.Module):
            """A lightweight LFCC + LCNN baseline used for quick reproducible experiments."""

            def __init__(self, n_lfcc: int = 60):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)


        LFCC_TRANSFORM = torchaudio.transforms.LFCC(
            sample_rate=16000,
            n_lfcc=60,
            speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 320},
        )


        class LFCCDataset(Dataset):
            """Dataset wrapper that computes LFCC features on demand from metadata rows."""

            def __init__(self, dataframe: pd.DataFrame, max_frames: int = 400):
                self.df = dataframe.reset_index(drop=True)
                self.max_frames = max_frames

            def __len__(self) -> int:
                return len(self.df)

            def __getitem__(self, index: int):
                row = self.df.iloc[index]
                wav, sr = torchaudio.load(row["audio_path"])
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                feat = LFCC_TRANSFORM(wav)
                if feat.shape[2] < self.max_frames:
                    feat = F.pad(feat, (0, self.max_frames - feat.shape[2]))
                else:
                    feat = feat[:, :, : self.max_frames]
                return feat, int(row["label"])


        def preprocess_audio_ref(audio_ref) -> torch.Tensor:
            """Load and normalize an audio reference into a mono 16 kHz waveform tensor."""
            if isinstance(audio_ref, str):
                wav, sr = torchaudio.load(audio_ref)
            elif isinstance(audio_ref, dict):
                wav = torch.from_numpy(audio_ref["array"]).float().unsqueeze(0)
                sr = audio_ref["sampling_rate"]
            else:
                raise TypeError(f"Unsupported audio reference type: {type(audio_ref)}")

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            return wav


        def prepare_aasist_input(wav: torch.Tensor, cut: int = 64600) -> torch.Tensor:
            """Crop or pad waveform for AASIST-family models."""
            if wav.shape[1] < cut:
                wav = F.pad(wav, (0, cut - wav.shape[1]))
            else:
                wav = wav[:, :cut]
            return wav


        def predict_aasist(model: nn.Module, audio_ref, device: str = DEVICE) -> float:
            """Return the bonafide score for AASIST and AASIST-L."""
            wav = prepare_aasist_input(preprocess_audio_ref(audio_ref)).to(device)
            with torch.no_grad():
                _, logits = model(wav)
            probs = logits.softmax(dim=1).cpu().numpy()[0]
            return float(probs[1])


        def predict_aasist3(model: nn.Module, audio_ref, device: str = DEVICE) -> float:
            """Return the bonafide score for the current AASIST3 checkpoint."""
            wav = prepare_aasist_input(preprocess_audio_ref(audio_ref)).to(device)
            with torch.no_grad():
                logits = model(wav)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return float(probs[0])


        def predict_lfcc_lcnn(model: nn.Module, audio_ref, device: str = DEVICE, max_frames: int = 400) -> float:
            """Return the bonafide score for LFCC + LCNN."""
            wav = preprocess_audio_ref(audio_ref)
            feat = LFCC_TRANSFORM(wav)
            if feat.shape[2] < max_frames:
                feat = F.pad(feat, (0, max_frames - feat.shape[2]))
            else:
                feat = feat[:, :, :max_frames]
            with torch.no_grad():
                logits = model(feat.unsqueeze(0).to(device))
            return float(torch.softmax(logits, dim=1)[0, 1].item())
        '''
    ),
    code(
        '''
        def ensure_aasist_repo(repo_dir: Path = Path("/kaggle/working/aasist")) -> Path:
            """Clone the AASIST repository only when it is missing."""
            if not repo_dir.exists():
                !git clone https://github.com/clovaai/aasist.git {repo_dir}
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            return repo_dir


        def ensure_aasist3_repo(repo_dir: Path = Path("/kaggle/working/AASIST3")) -> Path:
            """Clone the AASIST3 repository only when it is missing."""
            if not repo_dir.exists():
                !git clone https://github.com/mtuciru/AASIST3.git {repo_dir}
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
            return repo_dir


        def load_aasist_model(config_name: str, weight_name: str) -> nn.Module:
            """Load one AASIST-family checkpoint from the official repository."""
            repo_dir = ensure_aasist_repo()
            current_dir = Path.cwd()
            try:
                os.chdir(repo_dir)
                with open(repo_dir / "config" / config_name, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                module = importlib.import_module(f"models.{config['model_config']['architecture']}")
                model = module.Model(config["model_config"]).to(DEVICE)
                model.load_state_dict(torch.load(repo_dir / "models" / "weights" / weight_name, map_location=DEVICE))
                model.eval()
            finally:
                os.chdir(current_dir)
            return model


        def load_aasist3_model() -> nn.Module:
            """Load the AASIST3 checkpoint from Hugging Face through the repo wrapper."""
            repo_dir = ensure_aasist3_repo()
            current_dir = Path.cwd()
            try:
                os.chdir(repo_dir)
                from model import aasist3
                model = aasist3.from_pretrained("MTUCI/AASIST3").to(DEVICE).eval()
            finally:
                os.chdir(current_dir)
            return model


        def load_lfcc_checkpoint_if_available() -> tuple[nn.Module | None, Path | None]:
            """Load a saved LFCC + LCNN checkpoint if one already exists."""
            checkpoint_path = find_existing_file(
                "checkpoints",
                [
                    "lfcc_lcnn.pth",
                    "checkpoints/lfcc_lcnn.pth",
                    "lfcc_lcnn_checkpoint.pth",
                ],
            )
            if checkpoint_path is None:
                return None, None
            model = SimpleLCNN().to(DEVICE)
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            model.eval()
            return model, checkpoint_path


        MODEL_REGISTRY: dict[str, dict] = {}

        if "AASIST" in RUN_CONFIG["enable_models"]:
            MODEL_REGISTRY["AASIST"] = {
                "model": None,
                "loader_fn": lambda: load_aasist_model("AASIST.conf", "AASIST.pth"),
                "predict_fn": predict_aasist,
                "type": "lazy_pretrained",
            }

        if "AASIST-L" in RUN_CONFIG["enable_models"]:
            MODEL_REGISTRY["AASIST-L"] = {
                "model": None,
                "loader_fn": lambda: load_aasist_model("AASIST-L.conf", "AASIST-L.pth"),
                "predict_fn": predict_aasist,
                "type": "lazy_pretrained",
            }

        if "AASIST3" in RUN_CONFIG["enable_models"]:
            MODEL_REGISTRY["AASIST3"] = {
                "model": None,
                "loader_fn": load_aasist3_model,
                "predict_fn": predict_aasist3,
                "type": "lazy_pretrained",
            }

        if "LFCC+LCNN" in RUN_CONFIG["enable_models"]:
            MODEL_REGISTRY["LFCC+LCNN"] = {
                "model": None,
                "loader_fn": None,
                "predict_fn": predict_lfcc_lcnn,
                "type": "lazy_checkpoint_or_train",
                "checkpoint_path": None,
            }


        def ensure_model_ready(model_name: str, model_entry: dict) -> None:
            """Load or train a model only when fresh inference is required."""
            if model_entry.get("model") is not None:
                return

            if model_name == "LFCC+LCNN":
                model, checkpoint_path = load_lfcc_checkpoint_if_available()
                if model is not None and not RUN_CONFIG["force_train_lfcc"]:
                    model_entry["model"] = model
                    model_entry["type"] = "checkpoint"
                    model_entry["checkpoint_path"] = str(checkpoint_path)
                    return
                train_lfcc_lcnn_if_needed()
                return

            loader_fn = model_entry.get("loader_fn")
            if loader_fn is None:
                raise RuntimeError(f"No loader is registered for {model_name}.")
            model_entry["model"] = loader_fn()
            model_entry["type"] = "pretrained"


        print_header("Registered models")
        for name, entry in MODEL_REGISTRY.items():
            print(f"{name:12s} | {entry['type']}")
        '''
    ),
    md(
        """
        ## Part 3 — Training Only for Models Without a Checkpoint

        LFCC + LCNN is trained only when no checkpoint artifact can be found.
        """
    ),
    code(
        '''
        def train_lfcc_lcnn_if_needed() -> None:
            """Train LFCC + LCNN only when the checkpoint is missing or retraining is forced."""
            entry = MODEL_REGISTRY.get("LFCC+LCNN")
            if entry is None:
                return

            checkpoint_target = DATASET_OUTPUT_DIRS["checkpoints"] / "lfcc_lcnn.pth"
            summary_target = DATASET_OUTPUT_DIRS["checkpoints"] / "lfcc_lcnn_train_summary.json"

            if entry["model"] is not None and not RUN_CONFIG["force_train_lfcc"]:
                print(f"Using existing LFCC + LCNN checkpoint: {entry['checkpoint_path']}")
                return

            train_df = get_dataset_bundle("asv19")["train_df"]
            if train_df is None or train_df.empty:
                raise RuntimeError("ASVspoof 2019 train metadata is required to train LFCC + LCNN.")

            model = SimpleLCNN().to(DEVICE)
            train_dataset = LFCCDataset(train_df)
            train_loader = DataLoader(
                train_dataset,
                batch_size=RUN_CONFIG["lfcc_batch_size"],
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RUN_CONFIG["lfcc_epochs"])
            criterion = nn.CrossEntropyLoss()
            history = []

            print_header("Training LFCC + LCNN")
            for epoch in range(RUN_CONFIG["lfcc_epochs"]):
                model.train()
                total_loss = 0.0
                total_examples = 0
                total_correct = 0

                for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{RUN_CONFIG['lfcc_epochs']}"):
                    features = features.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item()) * labels.size(0)
                    total_examples += labels.size(0)
                    total_correct += int((logits.argmax(dim=1) == labels).sum().item())

                scheduler.step()
                epoch_summary = {
                    "epoch": epoch + 1,
                    "loss": total_loss / max(total_examples, 1),
                    "accuracy": total_correct / max(total_examples, 1),
                }
                history.append(epoch_summary)
                print(f"Epoch {epoch + 1}: loss={epoch_summary['loss']:.4f}, accuracy={epoch_summary['accuracy']:.4f}")

            model.eval()
            torch.save(model.state_dict(), ensure_parent(checkpoint_target))
            save_json(
                summary_target,
                {
                    "epochs": RUN_CONFIG["lfcc_epochs"],
                    "batch_size": RUN_CONFIG["lfcc_batch_size"],
                    "history": history,
                },
            )

            MODEL_REGISTRY["LFCC+LCNN"]["model"] = model
            MODEL_REGISTRY["LFCC+LCNN"]["type"] = "trained_now"
            MODEL_REGISTRY["LFCC+LCNN"]["checkpoint_path"] = str(checkpoint_target)
            print(f"Saved checkpoint to: {checkpoint_target}")

        '''
    ),
    md(
        """
        ## Part 4 — Shared Evaluation and Error Analysis Helpers

        The helpers below implement the artifact-first workflow:
        - check saved results first
        - load when available
        - run only when missing
        - save outputs immediately
        """
    ),
    code(
        '''
        def get_dataset_output_dir(dataset_key: str) -> Path:
            """Return the local output directory for one dataset."""
            path = DATASET_OUTPUT_DIRS[dataset_key]
            path.mkdir(parents=True, exist_ok=True)
            return path


        def get_analysis_output_dir(dataset_key: str) -> Path:
            """Return the local error analysis directory for one dataset."""
            path = get_dataset_output_dir(dataset_key) / "error_analysis"
            path.mkdir(parents=True, exist_ok=True)
            return path


        def load_legacy_asv19_model_result(model_name: str) -> dict | None:
            """Load one model result from the legacy ASVspoof 2019 pickle format."""
            candidates = []
            for legacy_name in LEGACY_ASV19_PICKLES:
                found = find_existing_file("asv19", [legacy_name, f"results/{legacy_name}"])
                if found is not None:
                    candidates.append(found)

            for path in candidates:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                if model_name not in payload:
                    continue
                data = payload[model_name]
                return {
                    "scores": np.asarray(data["scores"]),
                    "labels": np.asarray(data["labels"]),
                    "eer": float(data["eer"]),
                    "source": str(path),
                }
            return None


        def find_standard_result_file(dataset_key: str, model_name: str) -> Path | None:
            """Locate a standardized per-model score file."""
            safe_model = safe_name(model_name)
            if dataset_key == "asv19":
                candidates = [
                    f"scores_{safe_model}.npz",
                    f"results/scores_{safe_model}.npz",
                    f"scores/asv19_{safe_model}.npz",
                ]
            elif dataset_key == "asv21_df":
                candidates = [
                    f"scores_2021DF_{safe_model}.npz",
                    f"results/scores_2021DF_{safe_model}.npz",
                    f"scores/asv21_df_{safe_model}.npz",
                ]
            elif dataset_key == "asv5":
                candidates = [
                    f"scores_asvspoof5_{RUN_CONFIG['asv5_split']}_{safe_model}.npz",
                    f"results/scores_asvspoof5_{RUN_CONFIG['asv5_split']}_{safe_model}.npz",
                    f"scores/asv5_{RUN_CONFIG['asv5_split']}_{safe_model}.npz",
                ]
            else:
                candidates = [f"scores_{safe_model}.npz"]
            return find_existing_file(dataset_key, candidates)


        def load_saved_model_result(dataset_key: str, model_name: str) -> dict | None:
            """Load cached evaluation results if they already exist."""
            standard_path = find_standard_result_file(dataset_key, model_name)
            if standard_path is not None:
                npz = np.load(standard_path, allow_pickle=True)
                result = {
                    "scores": np.asarray(npz["scores"]),
                    "labels": np.asarray(npz["labels"]),
                    "eer": float(npz["eer"]),
                    "source": str(standard_path),
                }
                if "utt_ids" in npz:
                    result["utt_ids"] = np.asarray(npz["utt_ids"]).astype(str)
                return result

            if dataset_key == "asv19":
                return load_legacy_asv19_model_result(model_name)

            return None


        def save_model_result(dataset_key: str, model_name: str, result: dict) -> Path:
            """Save one model's scores for one dataset."""
            safe_model = safe_name(model_name)
            if dataset_key == "asv19":
                filename = f"scores_{safe_model}.npz"
            elif dataset_key == "asv21_df":
                filename = f"scores_2021DF_{safe_model}.npz"
            elif dataset_key == "asv5":
                filename = f"scores_asvspoof5_{RUN_CONFIG['asv5_split']}_{safe_model}.npz"
            else:
                filename = f"scores_{safe_model}.npz"

            path = get_dataset_output_dir(dataset_key) / filename
            payload = {
                "scores": np.asarray(result["scores"]),
                "labels": np.asarray(result["labels"]),
                "eer": float(result["eer"]),
            }
            if "utt_ids" in result:
                payload["utt_ids"] = np.asarray(result["utt_ids"]).astype(str)
            np.savez(ensure_parent(path), **payload)
            return path


        def get_audio_reference(row: pd.Series, bundle: dict):
            """Return either an audio path or a Hugging Face audio object for the current row."""
            if pd.notna(row.get("audio_path")) and row.get("audio_path") is not None:
                return row["audio_path"]

            hf_dataset = bundle.get("hf_dataset")
            if hf_dataset is None:
                raise RuntimeError("This dataset row has no audio_path and no Hugging Face dataset handle.")

            sample = hf_dataset[int(row["hf_index"])]
            audio_key = find_first_key(sample, ["audio", "path", "speech"], None)
            if audio_key is None:
                raise RuntimeError("Could not infer the audio column for an ASVspoof 5 Hugging Face sample.")
            return sample[audio_key]


        def align_rows_with_result(eval_df: pd.DataFrame, result: dict) -> pd.DataFrame:
            """Align metadata rows to a saved score file when utt_ids are available."""
            if "utt_ids" not in result:
                if len(eval_df) != len(result["scores"]):
                    if 0 < len(eval_df) - len(result["scores"]) <= 20:
                        aligned = eval_df.iloc[: len(result["scores"])].reset_index(drop=True)
                    else:
                        raise RuntimeError("Cannot align scores to metadata because utt_ids were not stored and lengths differ.")
                else:
                    aligned = eval_df.iloc[: len(result["scores"])].reset_index(drop=True)
            else:
                aligned = eval_df.set_index("utt_id").loc[result["utt_ids"]].reset_index()

            if "labels" in result:
                saved_labels = np.asarray(result["labels"]).astype(int)
                meta_labels = aligned["label"].astype(int).values[: len(saved_labels)]
                mismatches = int(np.sum(meta_labels != saved_labels))
                tolerance = 20 if "utt_ids" not in result else 0
                if mismatches > tolerance:
                    raise RuntimeError(
                        f"Saved labels do not match metadata order ({mismatches} mismatches). "
                        "Refusing unsafe score attachment."
                    )
                if mismatches:
                    print(f"Warning: tolerated {mismatches} label mismatches while reusing legacy order-based scores.")
            return aligned


        def evaluate_one_model(dataset_key: str, bundle: dict, model_name: str, model_entry: dict) -> dict:
            """Evaluate one model on one dataset, unless a saved artifact already exists."""
            cached = load_saved_model_result(dataset_key, model_name)
            if cached is not None and not RUN_CONFIG["force_eval"][dataset_key]:
                align_rows_with_result(bundle["eval_df"], cached)
                print(f"[{dataset_key}] Loaded cached result for {model_name} from {cached['source']}")
                return cached

            ensure_model_ready(model_name, model_entry)
            model = model_entry["model"]
            if model is None:
                raise RuntimeError(f"{model_name} is enabled but no model instance is available.")

            predict_fn = model_entry["predict_fn"]
            eval_df = bundle["eval_df"].reset_index(drop=True)
            scores = []
            labels = []
            utt_ids = []
            errors = 0
            start_time = time.time()

            for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"{dataset_key}:{model_name}"):
                try:
                    audio_ref = get_audio_reference(row, bundle)
                    score = predict_fn(model, audio_ref, DEVICE)
                    scores.append(score)
                    labels.append(int(row["label"]))
                    utt_ids.append(str(row["utt_id"]))
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"  Error on {row['utt_id']}: {type(exc).__name__}: {exc}")

            result = {
                "scores": np.asarray(scores),
                "labels": np.asarray(labels),
                "utt_ids": np.asarray(utt_ids),
                "eer": compute_eer(np.asarray(scores), np.asarray(labels)),
                "n_errors": errors,
                "elapsed_sec": time.time() - start_time,
                "source": "fresh_run",
            }
            save_path = save_model_result(dataset_key, model_name, result)
            print(f"[{dataset_key}] Saved {model_name} to {save_path}")
            return result


        def save_dataset_summary(dataset_key: str, summary_rows: list[dict]) -> Path:
            """Save one dataset-level summary table."""
            path = get_dataset_output_dir(dataset_key) / "summary.json"
            save_json(path, {"rows": summary_rows})
            return path


        def evaluate_dataset(dataset_key: str, bundle: dict) -> dict:
            """Evaluate every enabled model on one dataset and save a summary immediately."""
            print_header(f"Evaluating {bundle['name']}")
            results = {}
            summary_rows = []
            for model_name in RUN_CONFIG["enable_models"]:
                if model_name not in MODEL_REGISTRY:
                    continue
                result = evaluate_one_model(dataset_key, bundle, model_name, MODEL_REGISTRY[model_name])
                results[model_name] = result
                summary_rows.append(
                    {
                        "model": model_name,
                        "eer": float(result["eer"]),
                        "n_scores": int(len(result["scores"])),
                        "n_errors": int(result.get("n_errors", 0)),
                        "source": result.get("source", "unknown"),
                    }
                )
                print(f"{model_name:12s} | EER={result['eer']:.2f}% | N={len(result['scores']):,}")
            save_dataset_summary(dataset_key, summary_rows)
            return results


        def attach_scores(bundle: dict, dataset_results: dict) -> pd.DataFrame:
            """Attach all model scores to one metadata table."""
            base_df = bundle["eval_df"][COMMON_METADATA_COLUMNS].copy()
            for model_name, result in dataset_results.items():
                aligned = align_rows_with_result(bundle["eval_df"], result)
                score_df = pd.DataFrame(
                    {
                        "utt_id": aligned["utt_id"].astype(str).values,
                        f"score_{model_name}": np.asarray(result["scores"]),
                    }
                )
                base_df = base_df.merge(score_df, on="utt_id", how="inner")
            return base_df.reset_index(drop=True)


        def grouped_eer_table(df: pd.DataFrame, group_col: str, score_cols: list[str]) -> pd.DataFrame:
            """Compute grouped EER by combining each spoof subgroup with all bonafide utterances."""
            bonafide_df = df[df["label"] == 1]
            rows = []
            for group_name, spoof_df in df[df["label"] == 0].groupby(group_col):
                subset = pd.concat([bonafide_df, spoof_df], ignore_index=True)
                row = {"group": group_name, "n_spoof": int(len(spoof_df))}
                for score_col in score_cols:
                    row[score_col] = compute_eer(subset[score_col].values, subset["label"].values)
                rows.append(row)
            return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


        def save_grouped_analysis(dataset_key: str, group_col: str, table: pd.DataFrame) -> None:
            """Save grouped EER tables and plots."""
            analysis_dir = get_analysis_output_dir(dataset_key)
            csv_path = analysis_dir / f"eer_by_{group_col}.csv"
            txt_path = analysis_dir / f"eer_by_{group_col}.txt"
            png_path = analysis_dir / f"eer_by_{group_col}.png"
            table.to_csv(csv_path, index=False)

            lines = [f"EER by {group_col}"]
            lines.append("=" * 80)
            lines.append(table.to_string(index=False))
            save_text(txt_path, "\\n".join(lines) + "\\n")

            fig, ax = plt.subplots(figsize=(12, 5))
            model_cols = [col for col in table.columns if col.startswith("score_")]
            x = np.arange(len(table))
            width = 0.8 / max(len(model_cols), 1)
            for idx, score_col in enumerate(model_cols):
                ax.bar(x + idx * width, table[score_col], width=width, label=score_col.replace("score_", ""))
            ax.set_xticks(x + width * (max(len(model_cols), 1) - 1) / 2)
            ax.set_xticklabels(table["group"], rotation=45, ha="right")
            ax.set_ylabel("EER (%)")
            ax.set_title(f"{dataset_key}: EER by {group_col}")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)


        def compute_failure_overlap(df: pd.DataFrame, model_pairs: list[tuple[str, str]]) -> pd.DataFrame:
            """Measure how often two models fail on the same utterances."""
            rows = []
            for left, right in model_pairs:
                left_col = f"score_{left}"
                right_col = f"score_{right}"
                if left_col not in df.columns or right_col not in df.columns:
                    continue

                for model_name, score_col in [(left, left_col), (right, right_col)]:
                    fpr, tpr, thresholds = roc_curve(df["label"], df[score_col], pos_label=1)
                    fnr = 1 - tpr
                    idx = np.nanargmin(np.abs(fpr - fnr))
                    df[f"pred_{model_name}"] = (df[score_col] >= thresholds[idx]).astype(int)

                wrong_left = set(df.loc[df[f"pred_{left}"] != df["label"], "utt_id"].tolist())
                wrong_right = set(df.loc[df[f"pred_{right}"] != df["label"], "utt_id"].tolist())
                both_wrong = wrong_left & wrong_right

                rows.append(
                    {
                        "pair": f"{left} vs {right}",
                        "model_1": left,
                        "model_2": right,
                        "errors_model_1": len(wrong_left),
                        "errors_model_2": len(wrong_right),
                        "both_wrong": len(both_wrong),
                        "only_model_1": len(wrong_left - wrong_right),
                        "only_model_2": len(wrong_right - wrong_left),
                        "overlap_ratio_percent": 100 * len(both_wrong) / max(len(wrong_left), len(wrong_right), 1),
                    }
                )
            return pd.DataFrame(rows)


        def save_failure_overlap(dataset_key: str, df: pd.DataFrame, overlap_df: pd.DataFrame, model_pairs: list[tuple[str, str]]) -> None:
            """Save failure overlap tables and score-correlation plots."""
            analysis_dir = get_analysis_output_dir(dataset_key)
            overlap_df.to_csv(analysis_dir / "failure_overlap.csv", index=False)
            save_text(analysis_dir / "failure_overlap.txt", overlap_df.to_string(index=False) + "\\n")

            valid_pairs = [(left, right) for left, right in model_pairs if f"score_{left}" in df.columns and f"score_{right}" in df.columns]
            if not valid_pairs:
                return

            fig, axes = plt.subplots(1, len(valid_pairs), figsize=(7 * len(valid_pairs), 6))
            if len(valid_pairs) == 1:
                axes = [axes]

            for ax, (left, right) in zip(axes, valid_pairs):
                ax.scatter(df[f"score_{left}"], df[f"score_{right}"], c=df["label"], s=4, alpha=0.15, cmap="coolwarm")
                corr = np.corrcoef(df[f"score_{left}"], df[f"score_{right}"])[0, 1]
                ax.set_xlabel(f"{left} score")
                ax.set_ylabel(f"{right} score")
                ax.set_title(f"{left} vs {right} (r={corr:.3f})")
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(analysis_dir / "score_correlation.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


        def run_error_analysis(dataset_key: str, bundle: dict, dataset_results: dict) -> pd.DataFrame:
            """Run error analysis after one dataset has finished."""
            analysis_dir = get_analysis_output_dir(dataset_key)
            summary_path = analysis_dir / "analysis_summary.json"
            if summary_path.exists() and not RUN_CONFIG["force_error_analysis"][dataset_key]:
                print(f"[{dataset_key}] Reusing existing error analysis from {analysis_dir}")
                return pd.read_csv(analysis_dir / "attached_scores.csv")

            print_header(f"Error analysis for {bundle['name']}")
            attached_df = attach_scores(bundle, dataset_results)
            attached_df.to_csv(analysis_dir / "attached_scores.csv", index=False)

            score_cols = [column for column in attached_df.columns if column.startswith("score_")]
            saved_groups = []
            for group_col in bundle["analysis_groups"]:
                if group_col not in attached_df.columns:
                    continue
                if attached_df[group_col].nunique(dropna=False) <= 1:
                    continue
                table = grouped_eer_table(attached_df, group_col, score_cols)
                save_grouped_analysis(dataset_key, group_col, table)
                saved_groups.append(group_col)
                print(f"Saved grouped EER for {group_col}")

            pair_candidates = [("AASIST", "LFCC+LCNN"), ("AASIST", "AASIST-L"), ("AASIST", "AASIST3")]
            overlap_df = compute_failure_overlap(attached_df.copy(), pair_candidates)
            if not overlap_df.empty:
                save_failure_overlap(dataset_key, attached_df, overlap_df, pair_candidates)
                print("Saved failure overlap analysis")

            save_json(
                summary_path,
                {
                    "dataset": dataset_key,
                    "saved_groups": saved_groups,
                    "n_rows": len(attached_df),
                    "score_columns": score_cols,
                },
            )
            return attached_df
        '''
    ),
    md(
        """
        ## Part 5 — ASVspoof 2019 LA

        Evaluate every model on ASVspoof 2019 LA, save the results, then run error analysis immediately.
        """
    ),
    code(
        '''
        BUNDLE_ASV19 = get_dataset_bundle("asv19")
        RESULTS_ASV19 = evaluate_dataset("asv19", BUNDLE_ASV19)
        ANALYSIS_ASV19 = run_error_analysis("asv19", BUNDLE_ASV19, RESULTS_ASV19)
        '''
    ),
    md(
        """
        ## Part 6 — ASVspoof 2021 DF

        Evaluate every model on ASVspoof 2021 DF, save the results, then run error analysis immediately.
        """
    ),
    code(
        '''
        BUNDLE_ASV21_DF = get_dataset_bundle("asv21_df")
        RESULTS_ASV21_DF = evaluate_dataset("asv21_df", BUNDLE_ASV21_DF)
        ANALYSIS_ASV21_DF = run_error_analysis("asv21_df", BUNDLE_ASV21_DF, RESULTS_ASV21_DF)
        '''
    ),
    md(
        """
        ## Part 7 — ASVspoof 5 (2024)

        Evaluate every model on ASVspoof 5, save the results, then run error analysis immediately.
        """
    ),
    code(
        '''
        BUNDLE_ASV5 = get_dataset_bundle("asv5")
        RESULTS_ASV5 = evaluate_dataset("asv5", BUNDLE_ASV5)
        ANALYSIS_ASV5 = run_error_analysis("asv5", BUNDLE_ASV5, RESULTS_ASV5)
        '''
    ),
    md(
        """
        ## Part 8 — Cross-Dataset Summary

        Summarize how each model behaves across all benchmark datasets.
        """
    ),
    code(
        '''
        def load_dataset_summary(dataset_key: str) -> pd.DataFrame | None:
            """Load one saved dataset summary JSON as a DataFrame when available."""
            path = get_dataset_output_dir(dataset_key) / "summary.json"
            if not path.exists():
                print(f"Skipping missing summary for {dataset_key}: {path}")
                return None
            payload = load_json(path)
            return pd.DataFrame(payload["rows"])


        summary_frames = []
        summary_19 = load_dataset_summary("asv19")
        if summary_19 is not None:
            summary_frames.append(summary_19[["model", "eer"]].rename(columns={"eer": "asv19_eer"}))
        summary_21 = load_dataset_summary("asv21_df")
        if summary_21 is not None:
            summary_frames.append(summary_21[["model", "eer"]].rename(columns={"eer": "asv21_df_eer"}))
        summary_5 = load_dataset_summary("asv5")
        if summary_5 is not None:
            summary_frames.append(summary_5[["model", "eer"]].rename(columns={"eer": f"asv5_{RUN_CONFIG['asv5_split']}_eer"}))

        if not summary_frames:
            raise RuntimeError("No dataset summaries are available yet.")

        cross_dataset = summary_frames[0]
        for frame in summary_frames[1:]:
            cross_dataset = cross_dataset.merge(frame, on="model", how="outer")
        if "asv19_eer" in cross_dataset.columns and "asv21_df_eer" in cross_dataset.columns:
            cross_dataset["delta_21_minus_19"] = cross_dataset["asv21_df_eer"] - cross_dataset["asv19_eer"]
        asv5_col = f"asv5_{RUN_CONFIG['asv5_split']}_eer"
        if "asv19_eer" in cross_dataset.columns and asv5_col in cross_dataset.columns:
            cross_dataset["delta_5_minus_19"] = cross_dataset[asv5_col] - cross_dataset["asv19_eer"]

        display(cross_dataset.sort_values("asv19_eer", na_position="last"))

        cross_dataset_path = RESULTS_ROOT / "cross_dataset_summary.csv"
        cross_dataset.to_csv(cross_dataset_path, index=False)
        print(f"Saved cross-dataset summary to: {cross_dataset_path}")
        '''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")

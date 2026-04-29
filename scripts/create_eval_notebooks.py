import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


COMMON_SETUP = r'''
import gc
import importlib
import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import time
import base64
from pathlib import Path

import numpy as np
import pandas as pd

# A failed import can leave torch half-initialized in the notebook kernel.
# Clear that stale state before retrying imports in the same session.
_torch_mod = sys.modules.get("torch")
if _torch_mod is not None and not hasattr(_torch_mod, "Tensor"):
    for _name in list(sys.modules):
        if _name == "torch" or _name.startswith("torch."):
            del sys.modules[_name]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import soundfile as sf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile"])
    import soundfile as sf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compute_eer(scores, labels) -> float:
    """Compute EER in percent. Higher score means more bonafide."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) * 50.0)


def load_pickle_results(candidates: list[Path]) -> dict:
    """Load the first existing results.pkl candidate."""
    for path in candidates:
        if path.exists():
            print(f"loading existing results: {path}")
            with open(path, "rb") as handle:
                return pickle.load(handle)
    return {}


def save_results_pickle(results: dict, output_path: Path) -> None:
    """Persist the unified dataset results file after each model finishes."""
    slim = {}
    for model_name, result in results.items():
        slim[model_name] = {
            "eer": float(result["eer"]),
            "scores": np.asarray(result["scores"], dtype=np.float64),
            "labels": np.asarray(result["labels"], dtype=np.int64),
        }
    ensure_parent(output_path)
    with open(output_path, "wb") as handle:
        pickle.dump(slim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {output_path} with models={list(slim)}")


def partial_path(output_dir: Path, model_name: str) -> Path:
    safe = model_name.replace("+", "_").replace(" ", "_").replace("/", "_")
    return output_dir / f"{safe}.partial.npz"


def load_partial(output_dir: Path, model_name: str, fallback_dirs: list[Path] | None = None) -> dict | None:
    path = partial_path(output_dir, model_name)
    if not path.exists():
        for fb_dir in (fallback_dirs or []):
            candidate = partial_path(fb_dir, model_name)
            if candidate.exists():
                path = candidate
                print(f"{model_name}: resuming partial from input dataset: {path}")
                break
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "scores": data["scores"].astype(np.float64).tolist(),
        "labels": data["labels"].astype(np.int64).tolist(),
        "utt_ids": data["utt_ids"].astype(str).tolist(),
    }


def save_partial(output_dir: Path, model_name: str, scores: list, labels: list, utt_ids: list) -> None:
    path = partial_path(output_dir, model_name)
    np.savez(
        ensure_parent(path),
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        utt_ids=np.asarray(utt_ids, dtype=str),
    )


def clear_partial(output_dir: Path, model_name: str) -> None:
    path = partial_path(output_dir, model_name)
    if path.exists():
        path.unlink()


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''


MODEL_CODE = r'''
import gc
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_torch_mod = sys.modules.get("torch")
if _torch_mod is not None and not hasattr(_torch_mod, "Tensor"):
    for _name in list(sys.modules):
        if _name == "torch" or _name.startswith("torch."):
            del sys.modules[_name]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class SimpleLCNN(nn.Module):
    """Small LFCC+LCNN baseline. This must match the saved checkpoint architecture."""

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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


LFCC_TRANSFORM = torchaudio.transforms.LFCC(
    sample_rate=16000,
    n_lfcc=60,
    speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 320},
)


def run_shell(command: str) -> None:
    """Notebook-friendly shell runner."""
    print(command)
    rc = os.system(command)
    if rc != 0:
        raise RuntimeError(f"command failed with exit code {rc}: {command}")


def ensure_aasist_repo(repo_dir: Path = Path("/kaggle/working/aasist")) -> Path:
    """Clone clovaai/aasist only when missing."""
    if not repo_dir.exists():
        run_shell(f"git clone https://github.com/clovaai/aasist.git {repo_dir}")
    repo_str = str(repo_dir)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    return repo_dir


def ensure_aasist3_repo(repo_dir: Path = Path("/kaggle/working/AASIST3")) -> Path:
    """Clone mtuciru/AASIST3 only when missing."""
    if not repo_dir.exists():
        run_shell(f"git clone https://github.com/mtuciru/AASIST3.git {repo_dir}")
    repo_str = str(repo_dir)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    return repo_dir


def load_aasist_model(config_name: str, weight_name: str) -> nn.Module:
    """Load AASIST or AASIST-L from the official repository."""
    repo_dir = ensure_aasist_repo()
    cwd = Path.cwd()
    try:
        os.chdir(repo_dir)
        _reset_module_namespace(["models"])
        with open(repo_dir / "config" / config_name, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        module = importlib.import_module(f"models.{cfg['model_config']['architecture']}")
        model = module.Model(cfg["model_config"]).to(DEVICE)
        state = torch.load(repo_dir / "models" / "weights" / weight_name, map_location=DEVICE)
        model.load_state_dict(state)
        return model.eval()
    finally:
        os.chdir(cwd)


def load_aasist3_model() -> nn.Module:
    """Load AASIST3 from Hugging Face through the AASIST3 repository wrapper."""
    repo_dir = ensure_aasist3_repo()
    cwd = Path.cwd()
    try:
        os.chdir(repo_dir)
        _reset_module_namespace(["model"])
        from model import aasist3
        return aasist3.from_pretrained("MTUCI/AASIST3").to(DEVICE).eval()
    finally:
        os.chdir(cwd)


SSL_AASIST_REPO_URL = "https://github.com/TakHemlata/SSL_Anti-spoofing.git"
SSL_AASIST_REPO_DIR = Path("/kaggle/working/SSL_Anti-spoofing")
NES2NET_REPO_URL = "https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW.git"
NES2NET_REPO_DIR = Path("/kaggle/working/Nes2Net_ASVspoof_ITW")
XLSR_300M_URL = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"


def install_pkg(pip_name: str, import_name: str | None = None) -> None:
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def _swap_sys_path(repo_dir: Path) -> None:
    repo_str = str(repo_dir)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)


def ensure_ssl_aasist_repo() -> Path:
    """Clone TakHemlata/SSL_Anti-spoofing only when missing."""
    if not SSL_AASIST_REPO_DIR.exists():
        run_shell(f"git clone {SSL_AASIST_REPO_URL} {SSL_AASIST_REPO_DIR}")
    _swap_sys_path(SSL_AASIST_REPO_DIR)
    return SSL_AASIST_REPO_DIR


def ensure_nes2net_repo() -> Path:
    """Clone Liu-Tianchi/Nes2Net_ASVspoof_ITW only when missing."""
    if not NES2NET_REPO_DIR.exists():
        run_shell(f"git clone {NES2NET_REPO_URL} {NES2NET_REPO_DIR}")
    _swap_sys_path(NES2NET_REPO_DIR)
    return NES2NET_REPO_DIR


def ensure_repo_fairseq(repo_dir: Path) -> None:
    """Install the repository-pinned fairseq when available.

    The SSL repos were written against a specific fairseq snapshot. PyPI fairseq
    can import but still be ABI/API-incompatible on Kaggle, so prefer the bundled
    editable install and use PyPI only as a last resort.
    """
    candidates = [p for p in repo_dir.iterdir() if p.is_dir() and p.name.startswith("fairseq")]
    if candidates:
        fairseq_dir = candidates[0]
        patch_fairseq_for_python312(fairseq_dir)
        print(f"installing repo-pinned fairseq from {fairseq_dir}")
        _swap_sys_path(fairseq_dir)
        # Kaggle currently ships a recent pip that rejects old omegaconf metadata
        # used by this fairseq snapshot. Downgrading pip and using the legacy
        # resolver keeps the install reproducible without changing the repo code.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pip<24.1"])
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--use-deprecated=legacy-resolver",
            "-e",
            str(fairseq_dir),
        ])
        patch_fairseq_for_python312(fairseq_dir)
        patch_hydra_for_python312()
        _reset_module_namespace(["fairseq", "hydra", "omegaconf"])
        importlib.invalidate_caches()
        try:
            import fairseq  # noqa: F401
            print(f"fairseq import path: {fairseq.__file__}")
        except ImportError as exc:
            raise ImportError(
                f"fairseq install finished but import still failed. "
                f"fairseq_dir={fairseq_dir}, sys.path[0]={sys.path[0]}"
            ) from exc
        return

    install_pkg("fairseq")
    patch_hydra_for_python312()
    _reset_module_namespace(["fairseq", "hydra", "omegaconf"])
    importlib.invalidate_caches()
    import fairseq  # noqa: F401
    print(f"fairseq import path: {fairseq.__file__}")


def patch_fairseq_for_python312(fairseq_dir: Path) -> None:
    """Patch old fairseq dataclasses so they import on Python 3.11/3.12.

    The pinned fairseq snapshot uses mutable dataclass defaults such as
    `common: CommonConfig = CommonConfig()`. Newer Python rejects that pattern.
    For this notebook we only rewrite config defaults to `field(default_factory=...)`.
    """
    configs_path = fairseq_dir / "fairseq" / "dataclass" / "configs.py"
    if not configs_path.exists():
        return

    n_replacements = patch_dataclass_mutable_defaults(configs_path)
    if n_replacements:
        print(f"patched {n_replacements} fairseq dataclass defaults for Python 3.12: {configs_path}")
    patch_fairseq_hydra_init_for_default_factory(fairseq_dir)
    patch_fairseq_numpy_aliases(fairseq_dir)


def patch_fairseq_hydra_init_for_default_factory(fairseq_dir: Path) -> None:
    """Make fairseq hydra_init understand dataclass default_factory fields.

    After Python 3.12 compatibility patching, fields such as `common` no longer
    have `.default`; they have `.default_factory`. Old fairseq hydra_init only
    reads `.default`, so it passes dataclasses.MISSING into OmegaConf. This patch
    instantiates default_factory values before registering them with Hydra.
    """
    init_path = fairseq_dir / "fairseq" / "dataclass" / "initialize.py"
    if not init_path.exists():
        return
    text = init_path.read_text(encoding="utf-8")
    if "patched_py312_default_factory" in text:
        return

    if "from dataclasses import MISSING" not in text:
        text = "from dataclasses import MISSING\n" + text

    old = "v = FairseqConfig.__dataclass_fields__[k].default"
    new = (
        "field_info = FairseqConfig.__dataclass_fields__[k]\n"
        "        if field_info.default is not MISSING:\n"
        "            v = field_info.default\n"
        "        elif field_info.default_factory is not MISSING:\n"
        "            v = field_info.default_factory()\n"
        "        else:\n"
        "            v = MISSING"
    )
    if old in text:
        text = text.replace(old, new)
        text += "\n# patched_py312_default_factory\n"
        init_path.write_text(text, encoding="utf-8")
        print(f"patched fairseq hydra_init default_factory handling: {init_path}")


def patch_hydra_for_python312() -> None:
    """Patch hydra 1.0.x dataclass defaults installed as a fairseq dependency."""
    spec = importlib.util.find_spec("hydra")
    if spec is None or not spec.submodule_search_locations:
        return
    hydra_root = Path(list(spec.submodule_search_locations)[0])
    conf_path = hydra_root / "conf" / "__init__.py"
    if not conf_path.exists():
        return
    n_replacements = patch_dataclass_mutable_defaults(conf_path)
    if n_replacements:
        print(f"patched {n_replacements} hydra dataclass defaults for Python 3.12: {conf_path}")


def patch_dataclass_mutable_defaults(path: Path) -> int:
    """Rewrite common mutable dataclass defaults for Python 3.11/3.12 compatibility."""
    import re

    text = path.read_text(encoding="utf-8")
    original = text

    if "from dataclasses import" in text and "field" not in text.split("from dataclasses import", 1)[1].split("\n", 1)[0]:
        text = re.sub(
            r"^(from dataclasses import )([^\n]+)$",
            lambda m: m.group(1) + m.group(2).rstrip() + ", field",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    elif "from dataclasses import" not in text:
        text = "from dataclasses import field\n" + text

    n_total = 0
    # Pattern: name: SomeConfig = SomeConfig()
    obj_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<name>\w+):\s+(?P<cls>[\w.]+)\s*=\s*(?P=cls)\(\)\s*$",
        flags=re.MULTILINE,
    )
    text, n_obj = obj_pattern.subn(
        lambda m: (
            f"{m.group('indent')}{m.group('name')}: {m.group('cls')} = "
            f"field(default_factory={m.group('cls')})"
        ),
        text,
    )
    n_total += n_obj

    # Pattern: name: List[T] = [] / name: Dict[K, V] = {}
    list_pattern = re.compile(r"^(?P<indent>\s*)(?P<name>\w+):\s+List\[(?P<t>[^\]]+)\]\s*=\s*\[\]\s*$", flags=re.MULTILINE)
    text, n_list = list_pattern.subn(
        lambda m: f"{m.group('indent')}{m.group('name')}: List[{m.group('t')}] = field(default_factory=list)",
        text,
    )
    n_total += n_list

    dict_pattern = re.compile(r"^(?P<indent>\s*)(?P<name>\w+):\s+Dict\[(?P<t>[^\]]+)\]\s*=\s*\{\}\s*$", flags=re.MULTILINE)
    text, n_dict = dict_pattern.subn(
        lambda m: f"{m.group('indent')}{m.group('name')}: Dict[{m.group('t')}] = field(default_factory=dict)",
        text,
    )
    n_total += n_dict

    if text != original:
        path.write_text(text, encoding="utf-8")
    return n_total


def patch_fairseq_numpy_aliases(fairseq_dir: Path) -> None:
    """Replace removed numpy scalar type aliases across the fairseq bundle.

    NumPy 1.24 removed np.float, np.int, np.bool, np.complex, np.object, np.str.
    The pinned fairseq snapshot still uses these in several files (e.g. indexed_dataset.py).
    """
    import re
    _aliases = [
        (re.compile(r'\bnp\.float\b'), 'np.float64'),
        (re.compile(r'\bnp\.int\b'), 'np.int_'),
        (re.compile(r'\bnp\.bool\b'), 'np.bool_'),
        (re.compile(r'\bnp\.complex\b'), 'np.complex128'),
        (re.compile(r'\bnp\.object\b'), 'object'),
        (re.compile(r'\bnp\.str\b'), 'np.str_'),
    ]
    total = 0
    for py_file in fairseq_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        original = text
        for pattern, replacement in _aliases:
            text = pattern.sub(replacement, text)
        if text != original:
            py_file.write_text(text, encoding="utf-8")
            total += 1
    if total:
        print(f"patched numpy deprecated aliases in {total} fairseq files")


def find_or_download_xlsr_300m() -> Path:
    """Locate or download the fairseq XLS-R 300M base SSL checkpoint."""
    candidates = [
        Path("/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr/xlsr2_300m.pt"),
        Path("/kaggle/input/datasets/minhbhm/sdd-survey/ssl_models/xlsr2_300m.pt"),
        Path("/kaggle/input/sdd-survey/checkpoints/xlsr/xlsr2_300m.pt"),
        Path("/kaggle/input/xlsr-300m/xlsr2_300m.pt"),
        Path("/kaggle/working/xlsr2_300m.pt"),
    ]
    for path in candidates:
        if path.exists():
            print(f"found XLS-R 300M base SSL: {path}")
            return path
    target = Path("/kaggle/working/xlsr2_300m.pt")
    print(f"downloading XLS-R 300M (~1.2 GB) to {target}")
    run_shell(f"wget -q -O {target} {XLSR_300M_URL}")
    return target


def _link_xlsr_into(repo_dir: Path) -> None:
    """Both SSL_Anti-spoofing and Nes2Net hard-code ./xlsr2_300m.pt — point it at the cached file."""
    target = repo_dir / "xlsr2_300m.pt"
    if target.exists():
        return
    src = find_or_download_xlsr_300m()
    try:
        target.symlink_to(src)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy(src, target)


def find_xlsr_aasist_checkpoint() -> Path:
    """Locate TakHemlata's pretrained Wav2Vec2-XLSR+AASIST anti-spoofing weights."""
    bases = [
        Path("/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr_aasist"),
        Path("/kaggle/input/sdd-survey/checkpoints/xlsr_aasist"),
        Path("/kaggle/input/xlsr-aasist-antispoofing"),
    ]
    names = ["Best_LA_model_for_DF.pth", "Best_LA_model_for_LA.pth", "Best_LA_model_for_ITW.pth"]
    for base in bases:
        for name in names:
            if (base / name).exists():
                return base / name
        if base.exists():
            for path in base.glob("*.pth"):
                return path
    raise FileNotFoundError(
        "Wav2Vec2-XLSR+AASIST pretrained weights not found. "
        "Upload TakHemlata 'Best_LA_model_for_DF.pth' as a Kaggle dataset under "
        "/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr_aasist/."
    )


def find_xlsr_nes2net_checkpoint() -> Path:
    """Locate Liu-Tianchi's pretrained wav2vec2+Nes2Net-X anti-spoofing weights."""
    bases = [
        Path("/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/wav2vec2_nes2net"),
        Path("/kaggle/input/sdd-survey/checkpoints/wav2vec2_nes2net"),
        Path("/kaggle/input/wav2vec2-nes2net"),
        Path("/kaggle/input/nes2net-asvspoof-itw"),
    ]
    for base in bases:
        if base.exists():
            for pattern in ("*avg_ckpt*.pth", "*.pth", "*.pt"):
                for path in sorted(base.glob(pattern)):
                    return path
    raise FileNotFoundError(
        "wav2vec2+Nes2Net pretrained weights not found. "
        "Upload Liu-Tianchi's averaged checkpoint as a Kaggle dataset under "
        "/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/wav2vec2_nes2net/."
    )


def _strip_state_prefix(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _reset_module_namespace(prefixes: list[str]) -> None:
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def _load_required_state(model: nn.Module, state: dict, model_name: str) -> None:
    """Load a checkpoint and fail on any architecture mismatch.

    Survey EER values are only useful when the full pretrained checkpoint is
    loaded. Silent partial loading can look successful while leaving random
    layers in the model, so missing/unexpected keys are fatal.
    """
    missing, unexpected = model.load_state_dict(_strip_state_prefix(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{model_name} checkpoint does not match the model architecture. "
            f"missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )


def load_xlsr_aasist_model() -> nn.Module:
    """Load Wav2Vec2-XLSR (300M) + AASIST from TakHemlata/SSL_Anti-spoofing."""
    repo_dir = ensure_ssl_aasist_repo()
    ensure_repo_fairseq(repo_dir)
    _link_xlsr_into(repo_dir)
    cwd = Path.cwd()
    try:
        os.chdir(repo_dir)
        _reset_module_namespace(["model", "models"])
        from model import Model as XLSRAASIST  # type: ignore
        import argparse
        model = XLSRAASIST(argparse.Namespace(), DEVICE).to(DEVICE)
        ckpt = find_xlsr_aasist_checkpoint()
        print(f"loading Wav2Vec2-XLSR+AASIST weights: {ckpt}")
        state = torch.load(ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        _load_required_state(model, state, "XLS-R+AASIST")
        return model.eval()
    finally:
        os.chdir(cwd)


def load_xlsr_nes2net_model() -> nn.Module:
    """Load wav2vec2 (XLS-R 300M) + Nes2Net-X from Liu-Tianchi/Nes2Net_ASVspoof_ITW."""
    repo_dir = ensure_nes2net_repo()
    ensure_repo_fairseq(repo_dir)
    _link_xlsr_into(repo_dir)
    cwd = Path.cwd()
    try:
        os.chdir(repo_dir)
        _reset_module_namespace(["model_scripts"])
        from model_scripts.wav2vec2_Nes2Net_X import wav2vec2_Nes2Net_no_Res_w_allT  # type: ignore
        import argparse
        # Defaults from the repo training command:
        # --pool_func mean --SE_ratio 1 --Nes_ratio 8 8
        args = argparse.Namespace(
            n_output_logits=2,
            Nes_ratio=[8, 8],
            dilation=2,
            pool_func="mean",
            SE_ratio=1,
        )
        model = wav2vec2_Nes2Net_no_Res_w_allT(args, DEVICE).to(DEVICE)
        ckpt = find_xlsr_nes2net_checkpoint()
        print(f"loading wav2vec2+Nes2Net-X weights: {ckpt}")
        state = torch.load(ckpt, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        _load_required_state(model, state, "XLS-R+Nes2Net")
        return model.eval()
    finally:
        os.chdir(cwd)


@torch.no_grad()
def predict_ssl_pair_batch(model, waves: torch.Tensor) -> np.ndarray:
    """SSL models return 2-class logits ordered [spoof, bonafide]."""
    logits = model(waves.to(DEVICE))
    if not torch.is_tensor(logits) or logits.dim() != 2 or logits.shape[1] != 2:
        raise RuntimeError(f"Unexpected SSL model output shape/type: {type(logits)} {getattr(logits, 'shape', None)}")
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def find_lfcc_checkpoint() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/lfcc_lcnn/lfcc_lcnn.pth"),
        Path("/kaggle/input/sdd-survey/checkpoints/lfcc_lcnn/lfcc_lcnn.pth"),
        Path("/kaggle/working/checkpoints/lfcc_lcnn/lfcc_lcnn.pth"),
        Path("results/checkpoints/lfcc_lcnn/lfcc_lcnn.pth"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("LFCC+LCNN checkpoint was not found. Attach sdd-survey or copy lfcc_lcnn.pth.")


def load_lfcc_lcnn_model() -> nn.Module:
    model = SimpleLCNN().to(DEVICE)
    ckpt = find_lfcc_checkpoint()
    print(f"loading LFCC+LCNN checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.eval()


def load_audio_16k(audio_ref):
    """Load path, bytes, or HF audio dict into mono 16 kHz waveform with shape (1, T)."""
    if isinstance(audio_ref, (str, Path)):
        wav, sr = torchaudio.load(str(audio_ref))
    elif isinstance(audio_ref, bytes):
        array, sr = sf.read(io.BytesIO(audio_ref), dtype="float32")
        if array.ndim == 1:
            wav = torch.from_numpy(array).float().unsqueeze(0)
        else:
            wav = torch.from_numpy(array).float().T
    elif isinstance(audio_ref, dict) and "array" in audio_ref and "sampling_rate" in audio_ref:
        wav = torch.from_numpy(audio_ref["array"]).float().unsqueeze(0)
        sr = int(audio_ref["sampling_rate"])
    else:
        raise TypeError(f"unsupported audio ref: {type(audio_ref)}")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav


def prepare_waveform(wav: torch.Tensor, cut: int = 64600) -> torch.Tensor:
    if wav.shape[1] < cut:
        wav = F.pad(wav, (0, cut - wav.shape[1]))
    else:
        wav = wav[:, :cut]
    return wav.squeeze(0)


def prepare_lfcc(wav: torch.Tensor, max_frames: int = 400) -> torch.Tensor:
    feat = LFCC_TRANSFORM(wav)
    if feat.shape[2] < max_frames:
        feat = F.pad(feat, (0, max_frames - feat.shape[2]))
    else:
        feat = feat[:, :, :max_frames]
    return feat.squeeze(0)


class WaveformDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        try:
            wav = prepare_waveform(load_audio_16k(row["audio_path"]))
            return wav, int(row["label"]), str(row["utt_id"]), 0
        except Exception:
            return torch.zeros(64600), int(row["label"]), str(row["utt_id"]), 1


class LFCCDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        try:
            feat = prepare_lfcc(load_audio_16k(row["audio_path"]))
            return feat, int(row["label"]), str(row["utt_id"]), 0
        except Exception:
            return torch.zeros(60, 400), int(row["label"]), str(row["utt_id"]), 1


@torch.no_grad()
def predict_aasist_batch(model, waves: torch.Tensor) -> np.ndarray:
    """AASIST and AASIST-L output logits [spoof, bonafide]."""
    _, logits = model(waves.to(DEVICE))
    return logits.softmax(dim=1)[:, 1].detach().cpu().numpy()


@torch.no_grad()
def predict_aasist3_batch(model, waves: torch.Tensor) -> np.ndarray:
    """AASIST3 output logits [spoof, bonafide]. Use index 1 as bonafide score."""
    logits = model(waves.to(DEVICE))
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


@torch.no_grad()
def predict_lfcc_batch(model, feats: torch.Tensor) -> np.ndarray:
    if feats.dim() == 3:
        feats = feats.unsqueeze(1)
    logits = model(feats.to(DEVICE))
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


MODEL_REGISTRY = {
    "AASIST": {
        "loader": lambda: load_aasist_model("AASIST.conf", "AASIST.pth"),
        "dataset": WaveformDataset,
        "predict": predict_aasist_batch,
        "batch_size": 64,
    },
    "AASIST-L": {
        "loader": lambda: load_aasist_model("AASIST-L.conf", "AASIST-L.pth"),
        "dataset": WaveformDataset,
        "predict": predict_aasist_batch,
        "batch_size": 64,
    },
    "AASIST3": {
        "loader": load_aasist3_model,
        "dataset": WaveformDataset,
        "predict": predict_aasist3_batch,
        "batch_size": 64,
    },
    "LFCC+LCNN": {
        "loader": load_lfcc_lcnn_model,
        "dataset": LFCCDataset,
        "predict": predict_lfcc_batch,
        "batch_size": 128,
    },
    "XLS-R+AASIST": {
        "loader": load_xlsr_aasist_model,
        "dataset": WaveformDataset,
        "predict": predict_ssl_pair_batch,
        "batch_size": 8,
    },
    "XLS-R+Nes2Net": {
        "loader": load_xlsr_nes2net_model,
        "dataset": WaveformDataset,
        "predict": predict_ssl_pair_batch,
        "batch_size": 8,
    },
}


COMPOUND_GROUPS = [
    ("WaveformDataset", ["XLS-R+AASIST", "XLS-R+Nes2Net"]),
]
'''


EVAL_LOCAL_CODE = r'''
def evaluate_model_on_dataframe(
    model_name: str,
    df: pd.DataFrame,
    results: dict,
    output_pkl: Path,
    output_dir: Path,
    force_eval: bool = False,
    partial_save_every: int = 5000,
    num_workers: int = 2,
    partial_input_dirs: list[Path] | None = None,
) -> dict:
    """Evaluate one model on local files with partial resume and unified pkl writes."""
    if model_name in results and not force_eval:
        print(f"{model_name}: already present in results.pkl, skipping")
        return results[model_name]

    entry = MODEL_REGISTRY[model_name]
    partial = load_partial(output_dir, model_name, partial_input_dirs)
    scores, labels, utt_ids = [], [], []
    completed = set()
    if partial:
        scores = partial["scores"]
        labels = partial["labels"]
        utt_ids = partial["utt_ids"]
        completed = set(utt_ids)
        print(f"{model_name}: resume partial with {len(completed):,} completed utterances")

    todo_df = df[~df["utt_id"].astype(str).isin(completed)].reset_index(drop=True)
    print(f"{model_name}: todo={len(todo_df):,} already_done={len(completed):,}")

    model = entry["loader"]()
    dataset = entry["dataset"](todo_df)
    loader = DataLoader(
        dataset,
        batch_size=entry["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    errors = 0
    last_save = len(utt_ids)
    t0 = time.time()
    try:
        for batch_inputs, batch_labels, batch_utt_ids, batch_is_error in tqdm(loader, desc=model_name):
            batch_scores = entry["predict"](model, batch_inputs)
            ok_mask = batch_is_error.numpy() == 0
            errors += int((~ok_mask).sum())
            for i, ok in enumerate(ok_mask):
                if not ok:
                    continue
                scores.append(float(batch_scores[i]))
                labels.append(int(batch_labels[i].item()))
                utt_ids.append(str(batch_utt_ids[i]))

            if len(utt_ids) - last_save >= partial_save_every:
                save_partial(output_dir, model_name, scores, labels, utt_ids)
                last_save = len(utt_ids)
    except KeyboardInterrupt:
        save_partial(output_dir, model_name, scores, labels, utt_ids)
        print(f"{model_name}: interrupted; partial saved")
        raise
    finally:
        release_model(model)
        model = None

    eer = compute_eer(scores, labels)
    result = {
        "eer": eer,
        "scores": np.asarray(scores, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    results[model_name] = result
    save_results_pickle(results, output_pkl)
    clear_partial(output_dir, model_name)
    print(f"{model_name}: EER={eer:.4f}% N={len(scores):,} errors={errors:,} elapsed={(time.time()-t0)/60:.1f} min")
    return result


def evaluate_models_compound(
    model_names: list[str],
    df: pd.DataFrame,
    results: dict,
    output_pkl: Path,
    output_dir: Path,
    force_eval: dict[str, bool] | None = None,
    partial_save_every: int = 5000,
    num_workers: int = 2,
    partial_input_dirs: list[Path] | None = None,
) -> None:
    """Run multiple models in one pass over the same data loader.

    All listed models must share the same dataset class. Each model has its own
    partial.npz checkpoint and is added to results.pkl independently when finished.
    The compound runner saves audio I/O versus running models sequentially because
    each utterance is decoded once and each model in the group consumes the same
    batch tensor. The XLS-R front-ends inside SSL models are NOT shared because
    each pretrained checkpoint contains independently fine-tuned SSL weights.
    """
    force_eval = force_eval or {}
    pending = []
    for name in model_names:
        if name in results and not force_eval.get(name, False):
            print(f"{name}: already present in results.pkl, skipping")
        else:
            pending.append(name)
    if not pending:
        return

    dataset_classes = {MODEL_REGISTRY[n]["dataset"] for n in pending}
    if len(dataset_classes) > 1:
        raise ValueError(f"compound run requires identical dataset class, got {dataset_classes}")
    DatasetCls = dataset_classes.pop()

    partials = {}
    completed = {}
    for name in pending:
        partial = load_partial(output_dir, name, partial_input_dirs)
        if partial:
            partials[name] = partial
            completed[name] = set(partial["utt_ids"])
            print(f"{name}: resume partial with {len(partial['utt_ids']):,} completed")
        else:
            partials[name] = {"scores": [], "labels": [], "utt_ids": []}
            completed[name] = set()

    todo_mask = df["utt_id"].astype(str).map(
        lambda u: any(u not in completed[n] for n in pending)
    )
    todo_df = df[todo_mask].reset_index(drop=True)
    print(f"compound[{'+'.join(pending)}]: todo={len(todo_df):,}")

    models = {}
    for name in pending:
        print(f"loading {name}")
        models[name] = MODEL_REGISTRY[name]["loader"]()
    batch_size = min(MODEL_REGISTRY[n]["batch_size"] for n in pending)

    dataset = DatasetCls(todo_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    last_save = {n: len(partials[n]["utt_ids"]) for n in pending}
    errors = 0
    t0 = time.time()
    desc = "+".join(pending)
    try:
        for batch_inputs, batch_labels, batch_utt_ids, batch_is_error in tqdm(loader, desc=desc):
            ok_mask = batch_is_error.numpy() == 0
            errors += int((~ok_mask).sum())
            for name in pending:
                entry = MODEL_REGISTRY[name]
                batch_scores = entry["predict"](models[name], batch_inputs)
                for i, ok in enumerate(ok_mask):
                    if not ok:
                        continue
                    utt_id = str(batch_utt_ids[i])
                    if utt_id in completed[name]:
                        continue
                    partials[name]["scores"].append(float(batch_scores[i]))
                    partials[name]["labels"].append(int(batch_labels[i].item()))
                    partials[name]["utt_ids"].append(utt_id)
                    completed[name].add(utt_id)
                if len(partials[name]["utt_ids"]) - last_save[name] >= partial_save_every:
                    save_partial(output_dir, name, partials[name]["scores"], partials[name]["labels"], partials[name]["utt_ids"])
                    last_save[name] = len(partials[name]["utt_ids"])
    except KeyboardInterrupt:
        for name in pending:
            save_partial(output_dir, name, partials[name]["scores"], partials[name]["labels"], partials[name]["utt_ids"])
        print("compound: interrupted; partials saved")
        raise
    finally:
        for name in pending:
            release_model(models[name])
            models[name] = None

    for name in pending:
        eer = compute_eer(partials[name]["scores"], partials[name]["labels"])
        results[name] = {
            "eer": eer,
            "scores": np.asarray(partials[name]["scores"], dtype=np.float64),
            "labels": np.asarray(partials[name]["labels"], dtype=np.int64),
        }
        save_results_pickle(results, output_pkl)
        clear_partial(output_dir, name)
        print(f"{name}: EER={eer:.4f}% N={len(partials[name]['scores']):,}")
    print(f"compound[{desc}] elapsed={(time.time()-t0)/60:.1f} min, errors={errors:,}")


def run_eval_plan(
    enabled_models: list[str],
    df: pd.DataFrame,
    results: dict,
    output_pkl: Path,
    output_dir: Path,
    force_eval: dict[str, bool] | None = None,
    partial_save_every: int = 5000,
    num_workers: int = 2,
    compound_enabled: bool = False,
    partial_input_dirs: list[Path] | None = None,
) -> None:
    """Schedule single-model and compound runs for the configured ENABLED_MODELS.

    Compound mode is opt-in because loading two XLS-R models at once can exceed Kaggle
    T4/P100 memory. When disabled, every model runs sequentially with the same partial
    resume behavior. When enabled, models declared in COMPOUND_GROUPS share one loader.
    """
    force_eval = force_eval or {}
    grouped = set()
    if compound_enabled:
        for _tag, group in COMPOUND_GROUPS:
            members = [m for m in group if m in enabled_models]
            if len(members) >= 2:
                evaluate_models_compound(
                    model_names=members,
                    df=df,
                    results=results,
                    output_pkl=output_pkl,
                    output_dir=output_dir,
                    force_eval=force_eval,
                    partial_save_every=partial_save_every,
                    num_workers=num_workers,
                    partial_input_dirs=partial_input_dirs,
                )
                grouped.update(members)
    for name in enabled_models:
        if name in grouped:
            continue
        evaluate_model_on_dataframe(
            model_name=name,
            df=df,
            results=results,
            output_pkl=output_pkl,
            output_dir=output_dir,
            force_eval=force_eval.get(name, False),
            partial_save_every=partial_save_every,
            num_workers=num_workers,
            partial_input_dirs=partial_input_dirs,
        )
'''


ASV2019_DATASET_CODE = r'''
DATASET_KEY = "asvspoof19"
OUTPUT_DIR = Path("/kaggle/working/asvspoof19")
OUTPUT_PKL = OUTPUT_DIR / "results.pkl"
INPUT_RESULTS = [
    OUTPUT_PKL,
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof19/results.pkl"),
    Path("/kaggle/input/sdd-survey/asvspoof19/results.pkl"),
    Path("results/asvspoof19/results.pkl"),
]
FORCE_EVAL = {
    "AASIST": False,
    "AASIST-L": False,
    "AASIST3": False,
    "LFCC+LCNN": False,
    "XLS-R+AASIST": False,
    "XLS-R+Nes2Net": False,
}
ENABLED_MODELS = ["AASIST", "AASIST-L", "AASIST3", "LFCC+LCNN", "XLS-R+AASIST", "XLS-R+Nes2Net"]
SMOKE_TEST_N = None
PARTIAL_SAVE_EVERY = 5000
NUM_WORKERS = 2
RUN_COMPOUND_SSL_MODELS = False
# Directories to search for *.partial.npz when /kaggle/working/ was lost after session expiry.
# Upload the saved partial files as a Kaggle dataset and set the path here.
PARTIAL_INPUT_DIRS = [
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof19"),
    Path("/kaggle/input/sdd-survey/asvspoof19"),
    Path("/kaggle/input/datasets/minhbhm/sdd-partials-asv19"),
    Path("/kaggle/input/sdd-partials-asv19"),
]


def locate_asvspoof2019_la() -> dict:
    """Locate ASVspoof 2019 LA paths in the Kaggle input layout."""
    candidates = [
        Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"),
        Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA"),
        Path("/kaggle/input/asvpoof-2019-dataset/LA/LA"),
        Path("/kaggle/input/asvpoof-2019-dataset/LA"),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "LA").exists() and not (base / "ASVspoof2019_LA_eval").exists():
            base = base / "LA"
        train_flac = base / "ASVspoof2019_LA_train" / "flac"
        eval_flac = base / "ASVspoof2019_LA_eval" / "flac"
        proto_dir = base / "ASVspoof2019_LA_cm_protocols"
        train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
        eval_proto = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
        if train_flac.exists() and eval_flac.exists() and train_proto.exists() and eval_proto.exists():
            return {"base": base, "train_flac": train_flac, "eval_flac": eval_flac, "train_proto": train_proto, "eval_proto": eval_proto}
    raise FileNotFoundError("ASVspoof 2019 LA was not found. Attach awsaf49/asvpoof-2019-dataset.")


def parse_asvspoof2019_protocol(protocol_path: Path, audio_dir: Path, split: str) -> pd.DataFrame:
    """ASVspoof 2019 LA CM protocol: speaker, utt_id, _, attack, key."""
    rows = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            rows.append({
                "split": split,
                "speaker": parts[0],
                "utt_id": utt_id,
                "attack": "bonafide" if parts[3] == "-" else parts[3],
                "label": 1 if parts[4] == "bonafide" else 0,
                "audio_path": str(audio_dir / f"{utt_id}.flac"),
            })
    df = pd.DataFrame(rows)
    missing = ~df["audio_path"].map(lambda p: Path(p).exists())
    if missing.any():
        print(f"warning: dropping {int(missing.sum())} rows with missing audio")
        df = df[~missing].copy()
    return df.reset_index(drop=True)


loc = locate_asvspoof2019_la()
print(loc)
eval_df = parse_asvspoof2019_protocol(loc["eval_proto"], loc["eval_flac"], "eval")
if SMOKE_TEST_N is not None:
    eval_df = eval_df.iloc[:SMOKE_TEST_N].copy()
print(eval_df["label"].value_counts().rename({1: "bonafide", 0: "spoof"}))
print(f"ASVspoof 2019 LA eval rows: {len(eval_df):,}")

results = load_pickle_results(INPUT_RESULTS)
'''


ASV2021_DATASET_CODE = r'''
DATASET_KEY = "asvspoof21"
OUTPUT_DIR = Path("/kaggle/working/asvspoof21")
OUTPUT_PKL = OUTPUT_DIR / "results.pkl"
INPUT_RESULTS = [
    OUTPUT_PKL,
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof21/results.pkl"),
    Path("/kaggle/input/sdd-survey/asvspoof21/results.pkl"),
    Path("results/asvspoof21/results.pkl"),
]
FORCE_EVAL = {
    "AASIST": False,
    "AASIST-L": False,
    "AASIST3": False,
    "LFCC+LCNN": False,
    "XLS-R+AASIST": False,
    "XLS-R+Nes2Net": False,
}
ENABLED_MODELS = ["AASIST", "LFCC+LCNN", "AASIST3", "AASIST-L", "XLS-R+AASIST", "XLS-R+Nes2Net"]
SMOKE_TEST_N = None
PARTIAL_SAVE_EVERY = 10000
NUM_WORKERS = 2
RUN_COMPOUND_SSL_MODELS = False
PARTIAL_INPUT_DIRS = [
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof21"),
    Path("/kaggle/input/sdd-survey/asvspoof21"),
    Path("/kaggle/input/datasets/minhbhm/sdd-partials-asv21"),
    Path("/kaggle/input/sdd-partials-asv21"),
]


def locate_asvspoof2021_df() -> dict:
    """Locate ASVspoof 2021 DF eval audio parts and trial metadata."""
    bases = [
        Path("/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021"),
        Path("/kaggle/input/avsspoof-2021"),
        Path("/kaggle/input/asvspoof-2021"),
    ]
    base = next((p for p in bases if p.exists()), None)
    if base is None:
        raise FileNotFoundError("ASVspoof 2021 root was not found. Attach mohammedabdeldayem/avsspoof-2021.")

    audio_dirs = []
    for part in ["ASVspoof2021_DF_eval_part00", "ASVspoof2021_DF_eval_part01", "ASVspoof2021_DF_eval_part02"]:
        root = base / part
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root):
            if any(name.endswith(".flac") for name in files):
                audio_dirs.append(Path(dirpath))
                break

    trial = base / "DF-keys-full" / "keys" / "DF" / "CM" / "trial_metadata.txt"
    if not trial.exists():
        for dirpath, _, files in os.walk(base / "DF-keys-full"):
            for name in files:
                if "trial" in name.lower() and name.endswith(".txt"):
                    trial = Path(dirpath) / name
                    break
            if trial.exists():
                break
    if not audio_dirs or not trial.exists():
        raise FileNotFoundError("Could not locate ASVspoof 2021 DF audio parts or trial metadata.")
    return {"base": base, "audio_dirs": audio_dirs, "trial": trial}


def parse_asvspoof2021_df(trial_path: Path, audio_dirs: list[Path]) -> pd.DataFrame:
    """Parse ASVspoof 2021 DF. The CM key is field index 5."""
    audio_index = {}
    for audio_dir in audio_dirs:
        for name in os.listdir(audio_dir):
            if name.endswith(".flac"):
                audio_index[name[:-5]] = str(audio_dir / name)

    rows = []
    with open(trial_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            label_text = parts[5]
            if label_text not in ("bonafide", "spoof"):
                continue
            utt_id = parts[1]
            if utt_id not in audio_index:
                continue
            rows.append({
                "speaker": parts[0],
                "utt_id": utt_id,
                "codec": parts[2],
                "source": parts[3] if len(parts) > 3 else "unknown",
                "attack": parts[4] if len(parts) > 4 else "unknown",
                "vocoder": parts[8] if len(parts) > 8 else "unknown",
                "label": 1 if label_text == "bonafide" else 0,
                "audio_path": audio_index[utt_id],
            })
    return pd.DataFrame(rows).reset_index(drop=True)


loc = locate_asvspoof2021_df()
print(loc)
eval_df = parse_asvspoof2021_df(loc["trial"], loc["audio_dirs"])
if SMOKE_TEST_N is not None:
    eval_df = eval_df.iloc[:SMOKE_TEST_N].copy()
print(eval_df["label"].value_counts().rename({1: "bonafide", 0: "spoof"}))
print(f"ASVspoof 2021 DF eval rows with audio: {len(eval_df):,}")

results = load_pickle_results(INPUT_RESULTS)
'''


IN_THE_WILD_DATASET_CODE = r'''
DATASET_KEY = "in_the_wild"
OUTPUT_DIR = Path("/kaggle/working/in_the_wild")
OUTPUT_PKL = OUTPUT_DIR / "results.pkl"
INPUT_RESULTS = [
    OUTPUT_PKL,
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/in_the_wild/results.pkl"),
    Path("/kaggle/input/sdd-survey/in_the_wild/results.pkl"),
    Path("results/in_the_wild/results.pkl"),
]
FORCE_EVAL = {
    "AASIST": False,
    "AASIST-L": False,
    "AASIST3": False,
    "LFCC+LCNN": False,
    "XLS-R+AASIST": False,
    "XLS-R+Nes2Net": False,
}
ENABLED_MODELS = ["AASIST", "AASIST-L", "AASIST3", "LFCC+LCNN", "XLS-R+AASIST", "XLS-R+Nes2Net"]
SMOKE_TEST_N = None
PARTIAL_SAVE_EVERY = 5000
NUM_WORKERS = 2
RUN_COMPOUND_SSL_MODELS = False
PARTIAL_INPUT_DIRS = [
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/in_the_wild"),
    Path("/kaggle/input/sdd-survey/in_the_wild"),
    Path("/kaggle/input/datasets/minhbhm/sdd-partials-itw"),
    Path("/kaggle/input/sdd-partials-itw"),
]


def locate_in_the_wild() -> dict:
    """Locate the In-the-Wild deepfake dataset on Kaggle.

    The published dataset (Müller et al., 2022) ships meta.csv with columns
    {file, speaker, label} where label is 'bona-fide' or 'spoof', plus a
    flat directory of WAV files. The Kaggle slug primary path is
    /kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake.
    """
    bases = [
        Path("/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake"),
        Path("/kaggle/input/abdallamohamed312/in-the-wild-audio-deepfake"),
        Path("/kaggle/input/in-the-wild-audio-deepfake"),
        Path("/kaggle/input/release-in-the-wild"),
        Path("/kaggle/input/in-the-wild"),
    ]
    base = next((p for p in bases if p.exists()), None)
    if base is None:
        raise FileNotFoundError(
            "In-the-Wild dataset was not found. Attach abdallamohamed312/in-the-wild-audio-deepfake."
        )

    meta = None
    for name in ("meta.csv", "metadata.csv", "labels.csv"):
        matches = list(base.rglob(name))
        if matches:
            meta = matches[0]
            break
    if meta is None:
        raise FileNotFoundError(f"Could not find In-the-Wild metadata CSV under {base}.")

    audio_dir = None
    for name in ("release_in_the_wild", "in_the_wild", "audio", "wavs"):
        matches = [p for p in base.rglob(name) if p.is_dir()]
        if matches:
            audio_dir = matches[0]
            break
    if audio_dir is None:
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for p in base.rglob(ext):
                audio_dir = p.parent
                break
            if audio_dir is not None:
                break
    if audio_dir is None:
        raise FileNotFoundError(f"Could not locate In-the-Wild audio directory under {base}.")
    return {"base": base, "meta": meta, "audio_dir": audio_dir}


def parse_in_the_wild(meta_path: Path, audio_dir: Path) -> pd.DataFrame:
    """Parse meta.csv to a unified dataframe with bonafide=1, spoof=0."""
    df = pd.read_csv(meta_path)
    cols_lower = {c.lower(): c for c in df.columns}
    file_col = next((cols_lower[k] for k in ("file", "filename", "path", "audio") if k in cols_lower), df.columns[0])
    label_col = next((cols_lower[k] for k in ("label", "class", "key") if k in cols_lower), df.columns[-1])
    speaker_col = cols_lower.get("speaker")

    audio_index = {}
    for ext in ("*.wav", "*.flac", "*.mp3"):
        for path in audio_dir.rglob(ext):
            audio_index[str(path.relative_to(audio_dir)).replace("\\", "/")] = path
            audio_index[path.name] = path
            audio_index[path.stem] = path

    rows = []
    missing_count = 0
    for _, row in df.iterrows():
        fname = str(row[file_col])
        label_text = str(row[label_col]).strip().lower()
        is_bonafide = label_text in ("bona-fide", "bonafide", "bona_fide", "real", "genuine", "1")
        candidate = audio_index.get(fname) or audio_index.get(Path(fname).name) or audio_index.get(Path(fname).stem)
        if candidate is None:
            direct = audio_dir / fname
            candidate = direct if direct.exists() else None
        if candidate is None:
            missing_count += 1
            continue
        rows.append({
            "utt_id": Path(fname).stem,
            "speaker": str(row[speaker_col]) if speaker_col else "unknown",
            "label": 1 if is_bonafide else 0,
            "label_text": label_text,
            "audio_path": str(candidate),
        })
    if missing_count:
        print(f"warning: dropping {missing_count} In-the-Wild rows with missing audio")
    out = pd.DataFrame(rows).reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"Parsed zero In-the-Wild rows from {meta_path}.")
    return out


loc = locate_in_the_wild()
print(loc)
eval_df = parse_in_the_wild(loc["meta"], loc["audio_dir"])
if SMOKE_TEST_N is not None:
    eval_df = eval_df.iloc[:SMOKE_TEST_N].copy()
print(eval_df["label"].value_counts().rename({1: "bonafide", 0: "spoof"}))
print(f"In-the-Wild rows with audio: {len(eval_df):,}")

results = load_pickle_results(INPUT_RESULTS)
'''


ASV5_DATASET_CODE = r'''
DATASET_KEY = "asvspoof5"
OUTPUT_DIR = Path("/kaggle/working/asvspoof5")
OUTPUT_PKL = OUTPUT_DIR / "results.pkl"
INPUT_RESULTS = [
    OUTPUT_PKL,
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof5/results.pkl"),
    Path("/kaggle/input/sdd-survey/asvspoof5/results.pkl"),
    Path("results/asvspoof5/results.pkl"),
]

# ASVspoof 5 Track 1 is stand-alone countermeasure evaluation: bonafide vs spoof.
# Use "dev" first for a full-system trial, then "eval" for the official large run.
ASV5_SPLIT = "dev"       # "train", "dev", or "eval"
ASV5_TRACK = "track_1"   # this notebook evaluates Track 1 only
ASV5_SOURCE = "hf_tar"   # "hf_tar", "auto", "local", or "hf_webdataset"
SMOKE_TEST_N = None
HF_DEBUG_N = 30
FORCE_EVAL = {
    "AASIST": False,
    "LFCC+LCNN": False,
    "AASIST3": False,
    "AASIST-L": False,
    "XLS-R+AASIST": False,
    "XLS-R+Nes2Net": False,
}
ENABLED_MODELS = ["AASIST", "LFCC+LCNN", "AASIST3", "AASIST-L", "XLS-R+AASIST", "XLS-R+Nes2Net"]
PARTIAL_SAVE_EVERY = 5000
NUM_WORKERS = 2
RUN_COMPOUND_SSL_MODELS = False
PARTIAL_INPUT_DIRS = [
    Path("/kaggle/input/datasets/minhbhm/sdd-survey/asvspoof5"),
    Path("/kaggle/input/sdd-survey/asvspoof5"),
    Path("/kaggle/input/datasets/minhbhm/sdd-partials-asv5"),
    Path("/kaggle/input/sdd-partials-asv5"),
]


def asv5_prefix_for_split(split: str) -> str:
    return {"train": "T", "dev": "D", "eval": "E"}[split]


def locate_asvspoof5_local() -> dict:
    """Find ASVspoof 5 Track 1 protocol and extracted FLAC directory."""
    slugs = ["asvspoof5", "asvspoof-5", "asvspoof5-data", "asvspoof5-dev", "asvspoof5-eval"]
    roots = [Path("/kaggle/input/datasets")] + [Path("/kaggle/input")]
    proto_names = [
        f"ASVspoof5.{ASV5_SPLIT}.{ASV5_TRACK}.tsv",
        f"{ASV5_SPLIT}.{ASV5_TRACK}.tsv",
        f"{ASV5_SPLIT}.track_1.tsv",
        "ASVspoof5.train.tsv" if ASV5_SPLIT == "train" else "",
        "train.tsv" if ASV5_SPLIT == "train" else "",
    ]
    prefix = asv5_prefix_for_split(ASV5_SPLIT)
    audio_dir_names = [f"flac_{prefix}", f"flac_{ASV5_SPLIT}", ASV5_SPLIT]

    bases = []
    for root in roots:
        if not root.exists():
            continue
        for slug in slugs:
            candidate = root / slug
            if candidate.exists():
                bases.append(candidate)
        bases.extend([p for p in root.glob("*asvspoof*5*") if p.is_dir()])

    seen = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        proto = None
        for name in proto_names:
            if not name:
                continue
            matches = list(base.rglob(name))
            if matches:
                proto = matches[0]
                break
        if proto is None:
            matches = list(base.rglob(f"*{ASV5_SPLIT}*{ASV5_TRACK}*.tsv"))
            if not matches and ASV5_SPLIT == "train":
                matches = list(base.rglob("*train*.tsv"))
            proto = matches[0] if matches else None

        audio_dir = None
        for name in audio_dir_names:
            matches = [p for p in base.rglob(name) if p.is_dir()]
            if matches:
                audio_dir = matches[0]
                break
        if audio_dir is None:
            for p in base.rglob(f"flac_{prefix}*"):
                if p.is_dir():
                    audio_dir = p
                    break

        if proto and audio_dir:
            return {"base": base, "protocol": proto, "audio_dir": audio_dir}

    raise FileNotFoundError(
        "ASVspoof 5 local files were not found. Expected Track 1 TSV plus extracted flac_T/flac_D/flac_E."
    )


def parse_asvspoof5_track1_protocol(protocol_path: Path, audio_dir: Path) -> pd.DataFrame:
    """Parse ASVspoof 5 Track 1 metadata.

    Official Track 1 rows are space-separated:
    SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED
    ATTACK_TAG ATTACK_LABEL KEY TMP
    KEY is the CM label: bonafide or spoof.
    """
    audio_index = {}
    for path in audio_dir.rglob("*"):
        if path.is_file():
            audio_index[path.name] = str(path)
            audio_index[path.stem] = str(path)

    rows = []
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 10:
                speaker, utt_id, gender, codec, codec_q, codec_seed, attack_tag, attack_label, key, tmp = parts[:10]
            elif len(parts) >= 5:
                # Defensive fallback for simplified TSV exports.
                speaker, utt_id, gender, attack_label, key = parts[:5]
                codec, codec_q, codec_seed, attack_tag, tmp = "-", "-", "-", "-", "-"
            else:
                continue
            if key not in ("bonafide", "spoof"):
                continue
            rows.append({
                "speaker": speaker,
                "utt_id": utt_id,
                "gender": gender,
                "codec": "nocodec" if codec == "-" else codec,
                "codec_q": codec_q,
                "codec_seed": codec_seed,
                "attack_tag": "bonafide" if attack_tag == "-" else attack_tag,
                "attack": "bonafide" if attack_label == "bonafide" else attack_label,
                "label": 1 if key == "bonafide" else 0,
                "audio_path": audio_index.get(utt_id) or audio_index.get(f"{utt_id}.flac"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No Track 1 rows parsed from {protocol_path}")
    df = df[df["audio_path"].notna()].copy()
    missing = ~df["audio_path"].map(lambda p: Path(p).exists())
    if missing.any():
        print(f"warning: dropping {int(missing.sum())} ASVspoof 5 rows with missing audio")
        df = df[~missing].copy()
    return df.reset_index(drop=True)


def build_asvspoof5_local_df() -> pd.DataFrame:
    loc = locate_asvspoof5_local()
    print(loc)
    df = parse_asvspoof5_track1_protocol(loc["protocol"], loc["audio_dir"])
    if SMOKE_TEST_N is not None:
        df = df.iloc[:SMOKE_TEST_N].copy()
    print(df["label"].value_counts().rename({1: "bonafide", 0: "spoof"}))
    print(f"ASVspoof 5 {ASV5_SPLIT} Track 1 rows with audio: {len(df):,}")
    print(df[["utt_id", "codec", "attack_tag", "attack", "label", "audio_path"]].head())
    return df


if ASV5_SOURCE == "local":
    eval_df = build_asvspoof5_local_df()
    RESOLVED_ASV5_SOURCE = "local"
elif ASV5_SOURCE == "hf_webdataset":
    eval_df = None
    RESOLVED_ASV5_SOURCE = "hf_webdataset"
elif ASV5_SOURCE == "hf_tar":
    eval_df = None
    RESOLVED_ASV5_SOURCE = "hf_tar"
elif ASV5_SOURCE == "auto":
    try:
        eval_df = build_asvspoof5_local_df()
        RESOLVED_ASV5_SOURCE = "local"
    except FileNotFoundError as exc:
        print(f"local ASVspoof 5 not found; falling back to Hugging Face tar files: {exc}")
        eval_df = None
        RESOLVED_ASV5_SOURCE = "hf_tar"
else:
    raise ValueError(f"Unknown ASV5_SOURCE={ASV5_SOURCE}")

print(f"resolved ASVspoof 5 source: {RESOLVED_ASV5_SOURCE}")
results = load_pickle_results(INPUT_RESULTS)
'''


ASV5_HF_CODE = r'''
ASV5_HF_REPO_ID = "jungjee/asvspoof5"
ASV5_CACHE_DIR = Path("/kaggle/working/asvspoof5_hf_cache")
ASV5_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_hf_hub():
    """Import huggingface_hub, installing it when Kaggle image does not include it."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    return hf_hub_download


def hf_download_asv5_file(filename: str) -> Path:
    """Download one ASVspoof 5 file from the HF repo cache."""
    hf_hub_download = ensure_hf_hub()
    path = hf_hub_download(
        repo_id=ASV5_HF_REPO_ID,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(ASV5_CACHE_DIR),
    )
    return Path(path)


def hf_tar_names_for_split(split: str) -> list[str]:
    """Return ASVspoof 5 FLAC tar filenames for one split."""
    if split == "train":
        return [f"flac_T_{suffix}.tar" for suffix in ["aa", "ab", "ac", "ad", "ae"]]
    if split == "dev":
        return [f"flac_D_{suffix}.tar" for suffix in ["aa", "ab", "ac"]]
    if split == "eval":
        return [f"flac_E_{suffix}.tar" for suffix in ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj"]]
    raise ValueError(f"Unknown ASVspoof 5 split: {split}")


def protocol_member_candidates(split: str) -> list[str]:
    if split == "train":
        return ["ASVspoof5.train.tsv", "train.tsv"]
    return [f"ASVspoof5.{split}.track_1.tsv", f"{split}.track_1.tsv"]


def read_protocol_from_hf_tar(split: str) -> str:
    """Read Track 1 protocol text from ASVspoof5_protocols.tar."""
    proto_tar = hf_download_asv5_file("ASVspoof5_protocols.tar")
    wanted = protocol_member_candidates(split)
    with tarfile.open(proto_tar, "r:*") as tar:
        names = tar.getnames()
        for member in tar:
            base = Path(member.name).name
            if base in wanted:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                return extracted.read().decode("utf-8")
    raise FileNotFoundError(f"Could not find {wanted} in {proto_tar}. Members include: {names[:20]}")


def load_asv5_hf_tar_protocol(split: str) -> dict:
    """Load Track 1 protocol rows directly from the protocol tar file."""
    text = read_protocol_from_hf_tar(split)
    meta = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        speaker, utt_id, gender, codec, codec_q, codec_seed, attack_tag, attack_label, key, tmp = parts[:10]
        if key not in ("bonafide", "spoof"):
            continue
        meta[utt_id] = {
            "speaker": speaker,
            "utt_id": utt_id,
            "gender": gender,
            "codec": "nocodec" if codec == "-" else codec,
            "codec_q": codec_q,
            "codec_seed": codec_seed,
            "attack_tag": "bonafide" if attack_tag == "-" else attack_tag,
            "attack": "bonafide" if attack_label == "bonafide" else attack_label,
            "label": 1 if key == "bonafide" else 0,
        }
    if not meta:
        raise RuntimeError(f"Parsed zero ASVspoof 5 protocol rows for split={split}")
    print(f"HF tar protocol rows for {split} Track 1: {len(meta):,}")
    return meta


def iter_asv5_hf_tar_audio(protocol: dict):
    """Yield (utt_id, flac_bytes) from downloaded ASVspoof 5 tar shards."""
    for filename in hf_tar_names_for_split(ASV5_SPLIT):
        tar_path = hf_download_asv5_file(filename)
        print(f"streaming {filename}: {tar_path}")
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                utt_id = Path(member.name).name
                if utt_id.endswith(".flac"):
                    utt_id = utt_id[:-5]
                if utt_id not in protocol:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                yield utt_id, extracted.read()


def build_asv5_hf_tar_index(protocol: dict) -> pd.DataFrame:
    """Scan ASVspoof 5 HF tar shards and verify audio/protocol matching."""
    rows = []
    seen = set()
    for utt_id, _ in tqdm(iter_asv5_hf_tar_audio(protocol), desc="scan_asv5_hf_tar"):
        if utt_id in seen:
            continue
        seen.add(utt_id)
        meta = protocol[utt_id]
        rows.append({
            "utt_id": utt_id,
            "label": int(meta["label"]),
            "speaker": meta.get("speaker", "unknown"),
            "codec": meta.get("codec", "unknown"),
            "attack_tag": meta.get("attack_tag", "unknown"),
            "attack": meta.get("attack", "unknown"),
        })
        if SMOKE_TEST_N is not None and len(rows) >= SMOKE_TEST_N:
            break
        if SMOKE_TEST_N is None and len(rows) >= len(protocol):
            break
    df = pd.DataFrame(rows)
    print("ASVspoof 5 HF tar scan summary")
    print(f"  protocol rows      : {len(protocol):,}")
    print(f"  matched audio rows : {len(df):,}")
    if df.empty:
        raise RuntimeError("ASVspoof 5 HF tar scan matched zero audio rows.")
    print(df.head())
    return df


def predict_tar_batch(model, audio_items: list[tuple[str, bytes, int]], entry: dict):
    """Decode a list of tar FLAC byte payloads and run one model batch."""
    inputs, labels, utt_ids = [], [], []
    for utt_id, audio_bytes, label in audio_items:
        wav = load_audio_16k(audio_bytes)
        if entry["dataset"] is WaveformDataset:
            inputs.append(prepare_waveform(wav))
        else:
            inputs.append(prepare_lfcc(wav))
        labels.append(label)
        utt_ids.append(utt_id)
    batch = torch.stack(inputs)
    return entry["predict"](model, batch), labels, utt_ids


def evaluate_asv5_hf_tar_model(
    model_name: str,
    results: dict,
    output_pkl: Path,
    output_dir: Path,
    force_eval: bool = False,
    partial_save_every: int = 5000,
    partial_input_dirs: list[Path] | None = None,
):
    """Evaluate ASVspoof 5 by reading FLAC bytes directly from HF tar shards."""
    if model_name in results and not force_eval:
        print(f"{model_name}: already present in results.pkl, skipping")
        return results[model_name]

    protocol = ASV5_HF_TAR_PROTOCOL
    target_ids = set(ASV5_HF_TAR_INDEX["utt_id"].astype(str))
    entry = MODEL_REGISTRY[model_name]
    model = entry["loader"]()

    partial = load_partial(output_dir, model_name, partial_input_dirs)
    scores, labels, utt_ids = [], [], []
    completed = set()
    if partial:
        scores, labels, utt_ids = partial["scores"], partial["labels"], partial["utt_ids"]
        completed = set(utt_ids)
        print(f"{model_name}: resume partial with {len(completed):,} completed utterances")

    pending = []
    last_save = len(utt_ids)
    t0 = time.time()
    try:
        for utt_id, audio_bytes in tqdm(iter_asv5_hf_tar_audio(protocol), desc=f"{model_name}:hf_tar"):
            if utt_id not in target_ids or utt_id in completed:
                continue
            pending.append((utt_id, audio_bytes, int(protocol[utt_id]["label"])))
            if len(pending) >= entry["batch_size"]:
                batch_scores, batch_labels, batch_ids = predict_tar_batch(model, pending, entry)
                scores.extend(float(x) for x in batch_scores)
                labels.extend(int(x) for x in batch_labels)
                utt_ids.extend(str(x) for x in batch_ids)
                completed.update(batch_ids)
                pending = []

            if len(utt_ids) - last_save >= partial_save_every:
                save_partial(output_dir, model_name, scores, labels, utt_ids)
                last_save = len(utt_ids)
            if SMOKE_TEST_N is not None and len(utt_ids) >= SMOKE_TEST_N:
                break

        if pending:
            batch_scores, batch_labels, batch_ids = predict_tar_batch(model, pending, entry)
            scores.extend(float(x) for x in batch_scores)
            labels.extend(int(x) for x in batch_labels)
            utt_ids.extend(str(x) for x in batch_ids)
    except KeyboardInterrupt:
        save_partial(output_dir, model_name, scores, labels, utt_ids)
        print(f"{model_name}: interrupted; partial saved")
        raise
    finally:
        release_model(model)
        model = None

    if not scores:
        raise RuntimeError("No ASVspoof 5 HF tar scores were produced.")
    eer = compute_eer(scores, labels)
    result = {"eer": eer, "scores": np.asarray(scores, dtype=np.float64), "labels": np.asarray(labels, dtype=np.int64)}
    results[model_name] = result
    save_results_pickle(results, output_pkl)
    clear_partial(output_dir, model_name)
    print(f"{model_name}: EER={eer:.4f}% N={len(scores):,} elapsed={(time.time()-t0)/60:.1f} min")
    return result


def _decode_protocol_bytes(value) -> str:
    def maybe_b64_decode(text: str) -> str | None:
        padded = text + "=" * (-len(text) % 4)
        try:
            decoded = base64.b64decode(padded).decode("utf-8")
            if "\n" in decoded and " " in decoded:
                return decoded
        except Exception:
            return None
        return None

    if isinstance(value, bytes):
        text = value.decode("utf-8")
        if "\n" in text and " " in text:
            return text
        return maybe_b64_decode(text.strip()) or text
    if isinstance(value, str):
        text = value.strip()
        if "\n" in text and " " in text:
            return text
        # Hugging Face's viewer may expose binary protocol payloads as base64-like
        # strings. Decode defensively when the direct string is not parseable text.
        return maybe_b64_decode(text) or text
    raise TypeError(f"unsupported protocol payload type: {type(value)}")


def load_asv5_hf_protocol(split: str) -> dict:
    """Load Track 1 labels from the protocol sample in the HF WebDataset stream."""
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    stream = load_dataset("jungjee/asvspoof5", split="train", streaming=True)
    first = next(iter(stream))
    protocol_key = "train.tsv" if split == "train" else f"{split}.track_1.tsv"
    if protocol_key not in first:
        alt = f"ASVspoof5.{split}.track_1.tsv"
        protocol_key = alt if alt in first else protocol_key
    if protocol_key not in first and split == "train" and "ASVspoof5.train.tsv" in first:
        protocol_key = "ASVspoof5.train.tsv"
    if protocol_key not in first:
        raise KeyError(f"Could not find {protocol_key} in ASVspoof 5 HF protocol sample. Keys: {list(first)}")

    meta = {}
    for line in _decode_protocol_bytes(first[protocol_key]).splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        speaker, utt_id, gender, codec, codec_q, codec_seed, attack_tag, attack_label, key, tmp = parts[:10]
        if key not in ("bonafide", "spoof"):
            continue
        meta[utt_id] = {
            "speaker": speaker,
            "utt_id": utt_id,
            "gender": gender,
            "codec": "nocodec" if codec == "-" else codec,
            "attack_tag": "bonafide" if attack_tag == "-" else attack_tag,
            "attack": "bonafide" if attack_label == "bonafide" else attack_label,
            "label": 1 if key == "bonafide" else 0,
        }
    print(f"HF protocol rows for {split} Track 1: {len(meta):,}")
    if not meta:
        print("Protocol sample keys:", list(first.keys()))
        inspect_asv5_hf_stream(HF_DEBUG_N)
        raise RuntimeError("ASVspoof 5 HF protocol parsed zero rows. Send the printed protocol/sample keys.")
    return meta


def describe_hf_sample(sample: dict, max_value_chars: int = 120) -> dict:
    """Return a compact printable description of one HF WebDataset sample."""
    desc = {"__key__": sample.get("__key__"), "fields": {}}
    for key, value in sample.items():
        if key == "__key__":
            continue
        if isinstance(value, bytes):
            desc["fields"][key] = f"bytes[{len(value):,}]"
        elif isinstance(value, dict):
            desc["fields"][key] = f"dict keys={list(value.keys())}"
        elif isinstance(value, str):
            compact = value[:max_value_chars].replace("\n", "\\n")
            desc["fields"][key] = f"str[{len(value):,}] {compact}"
        else:
            desc["fields"][key] = type(value).__name__
    return desc


def inspect_asv5_hf_stream(n: int = 30) -> list[dict]:
    """Print the first HF samples so real keys and payload fields are visible."""
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    stream = load_dataset("jungjee/asvspoof5", split="train", streaming=True)
    rows = []
    print(f"Inspecting first {n} ASVspoof 5 HF stream samples")
    for i, sample in enumerate(stream):
        desc = describe_hf_sample(sample)
        rows.append(desc)
        print(f"[{i}] key={desc['__key__']} fields={desc['fields']}")
        if i + 1 >= n:
            break
    return rows


def hf_audio_ref(sample: dict):
    """Return (utt_id, audio_ref) from one HF WebDataset example when it is audio."""
    key = sample.get("__key__", "")
    if not isinstance(key, str) or "/" not in key:
        return None, None
    utt_id = key.split("/")[-1]
    for audio_key in ["flac", "audio", "wav", "mp3"]:
        if audio_key in sample and isinstance(sample[audio_key], (bytes, dict, str)):
            return utt_id, sample[audio_key]
    for audio_key, value in sample.items():
        if audio_key.startswith("__"):
            continue
        if isinstance(value, (bytes, dict, str)):
            return utt_id, value
    return None, None


def asv5_hf_candidate_ids(sample: dict) -> list[str]:
    """Generate possible protocol utterance IDs from a HF sample."""
    candidates = []
    key = sample.get("__key__", "")
    if isinstance(key, str) and key:
        candidates.extend([key, key.split("/")[-1]])
        last = key.split("/")[-1]
        candidates.extend([Path(last).stem, last.replace(".flac", "")])
    for field, value in sample.items():
        if field.startswith("__"):
            continue
        if isinstance(value, str):
            candidates.extend([value, Path(value).name, Path(value).stem])

    seen = set()
    out = []
    for item in candidates:
        item = str(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_asv5_hf_utt_id(sample: dict, protocol: dict) -> str | None:
    """Find the protocol utterance ID matching a HF sample."""
    for candidate in asv5_hf_candidate_ids(sample):
        if candidate in protocol:
            return candidate
    return None


def predict_hf_batch(model, model_name: str, audio_items: list[tuple[str, object, int]], entry: dict):
    inputs, labels, utt_ids = [], [], []
    for utt_id, audio_bytes, label in audio_items:
        wav = load_audio_16k(audio_bytes)
        if entry["dataset"] is WaveformDataset:
            inputs.append(prepare_waveform(wav))
        else:
            inputs.append(prepare_lfcc(wav))
        labels.append(label)
        utt_ids.append(utt_id)
    batch = torch.stack(inputs)
    return entry["predict"](model, batch), labels, utt_ids


def build_asv5_hf_index(protocol: dict) -> pd.DataFrame:
    """Scan the HF stream and verify that audio samples match protocol rows before inference."""
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    stream = load_dataset("jungjee/asvspoof5", split="train", streaming=True)
    split_letter = asv5_prefix_for_split(ASV5_SPLIT)
    rows = []
    seen_prefixes = {}
    unmatched_examples = []
    scanned = 0
    audio_like = 0

    for sample in tqdm(stream, desc="scan_asv5_hf"):
        scanned += 1
        key = sample.get("__key__", "")
        top_prefix = key.split("/")[0] if isinstance(key, str) and "/" in key else str(key)
        seen_prefixes[top_prefix] = seen_prefixes.get(top_prefix, 0) + 1

        _, audio = hf_audio_ref(sample)
        if audio is None:
            continue
        audio_like += 1

        # Prefer the expected flac_D/flac_E/flac_T prefix, but do not require it:
        # some HF loaders expose only the filename. The protocol match is authoritative.
        if isinstance(key, str) and f"flac_{split_letter}" not in key and "/" in key:
            continue

        utt_id = resolve_asv5_hf_utt_id(sample, protocol)
        if utt_id is None:
            if len(unmatched_examples) < 5:
                unmatched_examples.append(describe_hf_sample(sample))
        else:
            meta = protocol[utt_id]
            rows.append({
                "utt_id": utt_id,
                "label": int(meta["label"]),
                "speaker": meta.get("speaker", "unknown"),
                "codec": meta.get("codec", "unknown"),
                "attack_tag": meta.get("attack_tag", "unknown"),
                "attack": meta.get("attack", "unknown"),
                "hf_key": key,
            })

        if SMOKE_TEST_N is not None and len(rows) >= SMOKE_TEST_N:
            break
        if SMOKE_TEST_N is None and len(rows) >= len(protocol):
            break

    df = pd.DataFrame(rows)
    print("ASVspoof 5 HF scan summary")
    print(f"  scanned samples     : {scanned:,}")
    print(f"  audio-like samples  : {audio_like:,}")
    print(f"  protocol rows       : {len(protocol):,}")
    print(f"  matched audio rows  : {len(df):,}")
    print(f"  seen key prefixes   : {seen_prefixes}")
    if unmatched_examples:
        print("  unmatched audio examples:")
        for example in unmatched_examples:
            print(f"    key={example['__key__']} fields={example['fields']}")
    if df.empty:
        inspect_asv5_hf_stream(HF_DEBUG_N)
        raise RuntimeError(
            "ASVspoof 5 HF scan matched zero audio rows. "
            "Send the printed inspect/scan output so the matcher can be adjusted."
        )
    print(df.head())
    return df


def evaluate_asv5_hf_webdataset_model(
    model_name: str,
    results: dict,
    output_pkl: Path,
    output_dir: Path,
    force_eval: bool = False,
    partial_save_every: int = 5000,
    partial_input_dirs: list[Path] | None = None,
):
    """Evaluate ASVspoof 5 from the HF WebDataset stream. Local extracted files are faster."""
    if model_name in results and not force_eval:
        print(f"{model_name}: already present in results.pkl, skipping")
        return results[model_name]

    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        from datasets import load_dataset

    protocol = ASV5_HF_PROTOCOL
    hf_index = ASV5_HF_INDEX
    target_ids = set(hf_index["utt_id"].astype(str))
    entry = MODEL_REGISTRY[model_name]
    model = entry["loader"]()

    partial = load_partial(output_dir, model_name, partial_input_dirs)
    scores, labels, utt_ids = [], [], []
    completed = set()
    if partial:
        scores, labels, utt_ids = partial["scores"], partial["labels"], partial["utt_ids"]
        completed = set(utt_ids)
        print(f"{model_name}: resume partial with {len(completed):,} completed utterances")

    stream = load_dataset("jungjee/asvspoof5", split="train", streaming=True)
    pending = []
    last_save = len(utt_ids)
    t0 = time.time()
    try:
        for sample in tqdm(stream, desc=f"{model_name}:hf_stream"):
            _, audio = hf_audio_ref(sample)
            if audio is None:
                continue
            utt_id = resolve_asv5_hf_utt_id(sample, protocol)
            if utt_id is None or utt_id not in target_ids or utt_id in completed:
                continue
            pending.append((utt_id, audio, int(protocol[utt_id]["label"])))
            if len(pending) >= entry["batch_size"]:
                batch_scores, batch_labels, batch_ids = predict_hf_batch(model, model_name, pending, entry)
                scores.extend(float(x) for x in batch_scores)
                labels.extend(int(x) for x in batch_labels)
                utt_ids.extend(str(x) for x in batch_ids)
                completed.update(batch_ids)
                pending = []
            if len(utt_ids) - last_save >= partial_save_every:
                save_partial(output_dir, model_name, scores, labels, utt_ids)
                last_save = len(utt_ids)
            if SMOKE_TEST_N is not None and len(utt_ids) >= SMOKE_TEST_N:
                break
        if pending:
            batch_scores, batch_labels, batch_ids = predict_hf_batch(model, model_name, pending, entry)
            scores.extend(float(x) for x in batch_scores)
            labels.extend(int(x) for x in batch_labels)
            utt_ids.extend(str(x) for x in batch_ids)
    except KeyboardInterrupt:
        save_partial(output_dir, model_name, scores, labels, utt_ids)
        print(f"{model_name}: interrupted; partial saved")
        raise
    finally:
        release_model(model)
        model = None

    if not scores:
        raise RuntimeError("No ASVspoof 5 HF scores were produced. Check ASV5_HF_INDEX and stream matcher output.")

    eer = compute_eer(scores, labels)
    result = {"eer": eer, "scores": np.asarray(scores, dtype=np.float64), "labels": np.asarray(labels, dtype=np.int64)}
    results[model_name] = result
    save_results_pickle(results, output_pkl)
    clear_partial(output_dir, model_name)
    print(f"{model_name}: EER={eer:.4f}% N={len(scores):,} elapsed={(time.time()-t0)/60:.1f} min")
    return result
'''


RUN_LOCAL_CODE = r'''
run_eval_plan(
    enabled_models=ENABLED_MODELS,
    df=eval_df,
    results=results,
    output_pkl=OUTPUT_PKL,
    output_dir=OUTPUT_DIR,
    force_eval=FORCE_EVAL,
    partial_save_every=PARTIAL_SAVE_EVERY,
    num_workers=NUM_WORKERS,
    compound_enabled=RUN_COMPOUND_SSL_MODELS,
    partial_input_dirs=PARTIAL_INPUT_DIRS,
)

save_results_pickle(results, OUTPUT_PKL)
print("final summary")
for name, result in results.items():
    print(f"{name:20s} EER={result['eer']:.4f}% N={len(result['scores']):,}")
'''


RUN_ASV5_CODE = r'''
if RESOLVED_ASV5_SOURCE == "local":
    run_eval_plan(
        enabled_models=ENABLED_MODELS,
        df=eval_df,
        results=results,
        output_pkl=OUTPUT_PKL,
        output_dir=OUTPUT_DIR,
        force_eval=FORCE_EVAL,
        partial_save_every=PARTIAL_SAVE_EVERY,
        num_workers=NUM_WORKERS,
        compound_enabled=RUN_COMPOUND_SSL_MODELS,
        partial_input_dirs=PARTIAL_INPUT_DIRS,
    )
elif RESOLVED_ASV5_SOURCE == "hf_tar":
    ASV5_HF_TAR_PROTOCOL = load_asv5_hf_tar_protocol(ASV5_SPLIT)
    ASV5_HF_TAR_INDEX = build_asv5_hf_tar_index(ASV5_HF_TAR_PROTOCOL)
    for model_name in ENABLED_MODELS:
        evaluate_asv5_hf_tar_model(
            model_name=model_name,
            results=results,
            output_pkl=OUTPUT_PKL,
            output_dir=OUTPUT_DIR,
            force_eval=FORCE_EVAL.get(model_name, False),
            partial_save_every=PARTIAL_SAVE_EVERY,
            partial_input_dirs=PARTIAL_INPUT_DIRS,
        )
elif RESOLVED_ASV5_SOURCE == "hf_webdataset":
    ASV5_HF_PROTOCOL = load_asv5_hf_protocol(ASV5_SPLIT)
    inspect_asv5_hf_stream(HF_DEBUG_N)
    ASV5_HF_INDEX = build_asv5_hf_index(ASV5_HF_PROTOCOL)
    for model_name in ENABLED_MODELS:
        evaluate_asv5_hf_webdataset_model(
            model_name=model_name,
            results=results,
            output_pkl=OUTPUT_PKL,
            output_dir=OUTPUT_DIR,
            force_eval=FORCE_EVAL.get(model_name, False),
            partial_save_every=PARTIAL_SAVE_EVERY,
            partial_input_dirs=PARTIAL_INPUT_DIRS,
        )
else:
    raise ValueError(f"Unknown RESOLVED_ASV5_SOURCE={RESOLVED_ASV5_SOURCE}")

save_results_pickle(results, OUTPUT_PKL)
print("final summary")
for name, result in results.items():
    print(f"{name:20s} EER={result['eer']:.4f}% N={len(result['scores']):,}")
'''


def build_asv2019() -> list[dict]:
    return [
        md("""# Evaluate ASVspoof 2019 LA\n\nThis notebook evaluates ASVspoof 2019 LA eval with resume-safe partial checkpoints. It reuses existing `results.pkl` from `sdd-survey` when possible and forces AASIST3 by default because the previous score used the wrong class index."""),
        md("""## Dataset structure\n\nASVspoof 2019 LA has `ASVspoof2019_LA_train`, `ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval`, and `ASVspoof2019_LA_cm_protocols`. The CM protocol row is `speaker utt_id - attack key`; `key=bonafide` maps to label 1 and `key=spoof` maps to label 0. This notebook evaluates the LA eval FLAC files only."""),
        code(COMMON_SETUP),
        code(ASV2019_DATASET_CODE),
        code(MODEL_CODE),
        code(EVAL_LOCAL_CODE),
        code(RUN_LOCAL_CODE),
    ]


def build_asv2021() -> list[dict]:
    return [
        md("""# Evaluate ASVspoof 2021 DF\n\nThis notebook updates ASVspoof 2021 DF `results.pkl`. Existing AASIST and LFCC+LCNN entries are reused unless forced; AASIST3 and AASIST-L are evaluated when missing."""),
        md("""## Dataset structure\n\nThe Kaggle ASVspoof 2021 package stores DF eval audio in `ASVspoof2021_DF_eval_part00/01/02` and metadata in `DF-keys-full/.../trial_metadata.txt`. The label is field index 5, not the last field. This notebook follows the same DF parsing used by the earlier AASIST and LFCC+LCNN runs."""),
        code(COMMON_SETUP),
        code(ASV2021_DATASET_CODE),
        code(MODEL_CODE),
        code(EVAL_LOCAL_CODE),
        code(RUN_LOCAL_CODE),
    ]


def build_in_the_wild() -> list[dict]:
    return [
        md("""# Evaluate In-the-Wild deepfake dataset\n\nThis notebook evaluates the Müller et al. 2022 In-the-Wild dataset (celebrity / politician deepfakes scraped from social media). It is the standard cross-dataset generalization probe alongside ASVspoof 2021 DF: real-world noise, codec mixing, and unknown synthesis pipelines that none of the models trained on. SSL models run sequentially by default for GPU memory safety; `RUN_COMPOUND_SSL_MODELS=True` enables shared audio loading when enough VRAM is available."""),
        md("""## Dataset structure\n\nKaggle slug: `abdallamohamed312/in-the-wild-audio-deepfake`. The package contains `meta.csv` (columns: `file`, `speaker`, `label`) and a flat directory of audio files (typically WAV at 16 kHz). `label='bona-fide'` maps to 1 and any other value (including `'spoof'`) maps to 0. The locator scans for both `meta.csv` and the audio directory under several candidate paths so the notebook still works if the slug is mounted under a different alias."""),
        code(COMMON_SETUP),
        code(IN_THE_WILD_DATASET_CODE),
        code(MODEL_CODE),
        code(EVAL_LOCAL_CODE),
        code(RUN_LOCAL_CODE),
    ]


def build_asv5() -> list[dict]:
    return [
        md("""# Evaluate ASVspoof 5 Track 1\n\nThis notebook evaluates ASVspoof 5 Track 1, the stand-alone countermeasure task. It saves a unified `/kaggle/working/asvspoof5/results.pkl` after each model and keeps per-model partial checkpoints so long runs can resume after interruption."""),
        md("""## Dataset structure\n\nASVspoof 5 audio is 16 kHz FLAC. Track 1 metadata files are space-separated protocol files such as `ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`, and `ASVspoof5.eval.track_1.tsv`. Rows contain `SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`; `KEY` is the CM label (`bonafide` or `spoof`). The corresponding audio prefixes are `flac_T` for train, `flac_D` for dev, and `flac_E` for eval. Track 2 enrollment/trial files are SASV protocols and are intentionally not used here."""),
        code(COMMON_SETUP),
        code(ASV5_DATASET_CODE),
        code(MODEL_CODE),
        code(EVAL_LOCAL_CODE),
        code(ASV5_HF_CODE),
        code(RUN_ASV5_CODE),
    ]


SUMMARY_MD = """# Evaluation Notebooks Summary

This folder contains four standalone Kaggle notebooks for dataset-level evaluation.

## Shared result format

Each notebook writes a unified `results.pkl`:

```python
{
    "AASIST": {"eer": float, "scores": np.ndarray, "labels": np.ndarray},
    ...
}
```

`labels` use `1=bonafide` and `0=spoof`. `scores` are always bonafide probabilities, so EER uses bonafide as the positive class.

## Resume behavior

Each model writes `<model>.partial.npz` during inference with `scores`, `labels`, and `utt_ids`. When a run is restarted, the notebook skips completed utterance IDs. Once a model finishes, the notebook immediately updates the dataset-level `results.pkl` and removes that model's partial file.

## ASVspoof 2019 LA

Notebook: `eval_asvspoof_2019.ipynb`

Expected Kaggle input paths:

- `/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset`
- fallback `/kaggle/input/asvpoof-2019-dataset`

The LA CM protocol row is `speaker utt_id - attack key`. Evaluation uses `ASVspoof2019_LA_eval/flac` and `ASVspoof2019.LA.cm.eval.trl.txt`. Existing `sdd-survey/asvspoof19/results.pkl` is reused, but AASIST3 is forced by default because the old score used the wrong class index.

## ASVspoof 2021 DF

Notebook: `eval_asvspoof_2021.ipynb`

Expected Kaggle input paths:

- `/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021`
- fallback `/kaggle/input/avsspoof-2021`

DF eval audio is split across `ASVspoof2021_DF_eval_part00/01/02`. The CM key is field index 5 in `trial_metadata.txt`. Existing AASIST and LFCC+LCNN results are reused unless forced; missing AASIST3 and AASIST-L entries are evaluated and appended.

## ASVspoof 5 Track 1

Notebook: `eval_asvspoof_5.ipynb`

This notebook evaluates Track 1 only: stand-alone bonafide/spoof countermeasure scoring. Track 2 files are for SASV enrollment/trials and are not compatible with the current `results.pkl` CM score format.

Important ASVspoof 5 structure:

- `ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`, `ASVspoof5.eval.track_1.tsv` are space-separated Track 1 metadata.
- Track 1 columns are `SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`.
- `KEY` is the binary CM label: `bonafide` -> 1, `spoof` -> 0.
- Audio prefixes are `flac_T` for train, `flac_D` for dev, and `flac_E` for eval.
- The Hugging Face mirror `jungjee/asvspoof5` is a WebDataset-style package with protocol files and large FLAC tar shards. `eval_asvspoof_5.ipynb` defaults to `ASV5_SOURCE='auto'`: it uses local extracted files if present, otherwise falls back to Hugging Face streaming. Kaggle Internet must be enabled for this fallback.
- In HF mode, the notebook now prints `inspect_asv5_hf_stream()` output and builds `ASV5_HF_INDEX` before loading any model. If matched audio rows are zero, it stops early with the printed keys/fields instead of running inference and failing at EER computation.
- Because Hugging Face `datasets` does not expose ASVspoof 5 extensionless FLAC payloads reliably, the current default is `ASV5_SOURCE='hf_tar'`. It downloads `ASVspoof5_protocols.tar` and the needed `flac_D_*.tar` or `flac_E_*.tar` shards with `huggingface_hub`, then streams FLAC bytes directly with Python `tarfile`.

## In-the-Wild

Notebook: `eval_in_the_wild.ipynb`

Müller et al. 2022 deepfake-in-the-wild dataset (~31K utterances) collected from social media. Standard cross-dataset generalization probe: none of the models were trained on it.

Expected Kaggle input paths:

- `/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake`
- fallback `/kaggle/input/in-the-wild-audio-deepfake`

The locator searches for `meta.csv` (columns `file`, `speaker`, `label`) and a flat audio directory. `label='bona-fide'` maps to 1, anything else to 0.

## Model score convention

AASIST, AASIST-L, AASIST3, XLS-R+AASIST, XLS-R+Nes2Net all produce two-class logits ordered as `[spoof, bonafide]` in these notebooks. The saved score is therefore:

```python
torch.softmax(logits, dim=1)[:, 1]
```

This is especially important for AASIST3. Using index 0 inverts the bonafide score and causes the ASVspoof 2019 EER to look wrong.

## SSL model additions: research findings, decisions, and code changes

This section documents what changed when the two SSL-based models (`XLS-R+AASIST`, `XLS-R+Nes2Net`) and the In-the-Wild notebook were added in the 2026-04 update.

### Research findings (web search, 2026-04)

The original survey only covered hand-crafted (`LFCC+LCNN`) and end-to-end graph-attention models (`AASIST`, `AASIST-L`, `AASIST3`). According to the broader literature (Tak et al. 2022, Stourbe et al. 2024, Liu et al. 2025) the dominant SOTA paradigm is now SSL front-end + lightweight back-end:

- WavLM-Large or XLS-R as the SSL front-end (each ~300M parameters).
- A small back-end (AASIST graph attention, MHFA, or Nes2Net nested Res2Net).
- WavLM is best on clean ASVspoof 2019 LA, XLS-R is reported best on ASVspoof DF, HuBERT is best on In-the-Wild.

In-the-Wild itself was already on `todo.md` as a generalization probe but had not been wired into a notebook. ASVspoof 2021 DF, ASVspoof 5, and ASVspoof 2019 LA covered codec compression and standard CM testing but not real-world social-media audio.

### Decisions

1. **Two new models** are added to the survey:
   - `XLS-R+AASIST` — Tak et al. 2022 (Odyssey), repo `TakHemlata/SSL_Anti-spoofing`, weights `Best_LA_model_for_DF.pth`.
   - `XLS-R+Nes2Net` — Liu et al. 2025, repo `Liu-Tianchi/Nes2Net_ASVspoof_ITW`, model class `wav2vec2_Nes2Net_no_Res_w_allT`. Reports 1.49% EER on ASVspoof 2021 DF and 5.52% EER on In-the-Wild.

   Both share the same XLS-R 300M front-end so the comparison isolates the back-end (AASIST graph attention vs Nes2Net nested Res2Net), which is one direct comparison the survey wanted.

   `WavLM+Nes2Net` and `WavLM+AASIST` were considered but rejected:
   - The `Liu-Tianchi/Nes2Net` main repo only ships `WavLM_Nes2Net*` checkpoints trained on the CtrSVDD singing-voice deepfake set, which is the wrong domain for ASVspoof / In-the-Wild evaluation.
   - No widely-released `WavLM+AASIST` pretrained weights for speech anti-spoofing exist.

2. **One new dataset notebook**: In-the-Wild (Kaggle slug `abdallamohamed312/in-the-wild-audio-deepfake`).

3. **Compound runner (option Z)** is available but disabled by default. When enabled, several models that share the same dataset class run in one pass over the data loader so each utterance is decoded only once. It stays opt-in because loading two XLS-R 300M models simultaneously can exceed Kaggle T4/P100 memory. The XLS-R front-ends inside the two SSL models are NOT shared because each pretrained checkpoint contains independently fine-tuned SSL weights (TakHemlata trained end-to-end; Liu-Tianchi used `lr=2.5e-7` which is consistent with SSL fine-tuning, and the README averaged checkpoint name encodes that learning rate). Sharing forward output would silently feed the wrong SSL features into the second back-end. The realised win is therefore audio I/O, not front-end compute.

   Per-batch audio caching (option Y) was rejected because XLS-R features at float16 are ~414 KB per utterance, so caching ASVspoof 2021 DF (~190 GB) and ASVspoof 5 eval (~280 GB) exceeds the 70 GB Kaggle working disk.

### Code changes in `scripts/create_eval_notebooks.py`

- `MODEL_CODE`:
  - Added `install_pkg`, `ensure_ssl_aasist_repo`, `ensure_nes2net_repo`, `find_or_download_xlsr_300m`, `_link_xlsr_into`, `find_xlsr_aasist_checkpoint`, `find_xlsr_nes2net_checkpoint`, `_strip_state_prefix`, `_reset_module_namespace`, `load_xlsr_aasist_model`, `load_xlsr_nes2net_model`, `predict_ssl_pair_batch`.
  - Registered `XLS-R+AASIST` and `XLS-R+Nes2Net` in `MODEL_REGISTRY` (both with `WaveformDataset`, batch size 8).
  - Added `COMPOUND_GROUPS` declaring `[XLS-R+AASIST, XLS-R+Nes2Net]` as a shared-loader group.
  - Added safety checks: repository-pinned fairseq is preferred over PyPI fairseq; SSL checkpoints must load with no missing/unexpected keys; generic module namespaces are reset before switching between cloned repos.
  - Nes2Net args follow the repo defaults used by the training command: `n_output_logits=2`, `dilation=2`, `pool_func='mean'`, `SE_ratio=1`, `Nes_ratio=[8, 8]`.
  - `predict_ssl_pair_batch` now expects a single `[batch, 2]` logits tensor and raises immediately on any unexpected output shape.

- `EVAL_LOCAL_CODE`:
  - Added `evaluate_models_compound` that loads multiple models simultaneously, iterates the loader once, runs each model on every batch, and saves each model's partial / pickle independently.
  - Added `run_eval_plan` that schedules compound and single-model runs based on `ENABLED_MODELS`, `COMPOUND_GROUPS`, and `RUN_COMPOUND_SSL_MODELS`. The default is sequential execution for VRAM safety.

- `RUN_LOCAL_CODE` and the `local` branch of `RUN_ASV5_CODE` switched to call `run_eval_plan(...)` instead of looping `evaluate_model_on_dataframe`.

- `ASV2019_DATASET_CODE`, `ASV2021_DATASET_CODE`, `ASV5_DATASET_CODE` now include `XLS-R+AASIST` and `XLS-R+Nes2Net` in `ENABLED_MODELS` and `FORCE_EVAL`.

- New constants `IN_THE_WILD_DATASET_CODE` (locator + parser for `meta.csv` schema) and `build_in_the_wild()` produce a fourth notebook `eval_in_the_wild.ipynb`. The parser builds a recursive audio index by relative path, filename, and stem to tolerate Kaggle layout differences. `main()` writes it alongside the existing three.

### Review fixes after adding SSL models

The first implementation was intentionally reviewed before running full experiments. The following issues were fixed:

- Partial checkpoint loading for SSL models was removed. Missing or unexpected checkpoint keys now raise an error, because partial loads can silently produce invalid EER values.
- The fairseq dependency now prefers each repo's pinned `fairseq-*` folder. PyPI fairseq is only a fallback. Kaggle's current pip rejects old `omegaconf` metadata, so the installer downgrades to `pip<24.1` and uses pip's legacy resolver for the pinned fairseq editable install.
- The pinned fairseq source directory is also inserted into `sys.path`, and the notebook verifies `import fairseq` immediately after install. This handles Kaggle sessions where editable install succeeds but the current kernel does not refresh module resolution.
- Old fairseq dataclass config defaults are patched automatically for Python 3.12 by rewriting mutable defaults like `common: CommonConfig = CommonConfig()` to `field(default_factory=CommonConfig)`.
- Hydra 1.0.x is installed as a fairseq dependency and has the same Python 3.12 dataclass issue. The notebook patches `hydra/conf/__init__.py` before importing fairseq, then clears `fairseq`, `hydra`, and `omegaconf` from `sys.modules` for a clean import.
- Because fairseq's old `hydra_init()` reads dataclass `.default`, it also needs a small patch after converting fields to `default_factory`; the notebook now instantiates `field_info.default_factory()` before registering configs with Hydra.
- The notebook prints `fairseq.__file__` after import so fairseq namespace conflicts between SSL repos are visible. SSL models still run sequentially by default, which is the safer path for both VRAM and fairseq import stability.
- Compound XLS-R evaluation is disabled by default through `RUN_COMPOUND_SSL_MODELS=False` to avoid T4/P100 out-of-memory failures.
- Generic Python module namespaces (`model`, `models`, `model_scripts`) are cleared before switching cloned repos to avoid stale imports.
- In-the-Wild audio discovery now uses a recursive audio index instead of assuming all files are directly under one flat directory.

### Required Kaggle input paths for the SSL models

- `/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr_aasist/Best_LA_model_for_DF.pth` — TakHemlata pretrained anti-spoofing weights (Google Drive link in their README).
- `/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/wav2vec2_nes2net/<file>.pth` — Liu-Tianchi pretrained averaged checkpoint, e.g. `wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.pth`.
- `xlsr2_300m.pt` is auto-downloaded from `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt` if not pre-staged. `fairseq` is `pip install`-ed at runtime.

### References used for dataset structure

- ASVspoof 5 Hugging Face dataset card: https://huggingface.co/datasets/jungjee/asvspoof5
- ASVspoof 5 README on Hugging Face, including Track 1 metadata columns and FLAC counts: https://huggingface.co/datasets/jungjee/asvspoof5/blob/main/README.txt
- TakHemlata/SSL_Anti-spoofing (XLS-R+AASIST baseline, weights via Google Drive): https://github.com/TakHemlata/SSL_Anti-spoofing
- Liu-Tianchi/Nes2Net_ASVspoof_ITW (XLS-R+Nes2Net-X, weights via Google Drive): https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW
- Nes2Net paper (arXiv 2504.05657): https://arxiv.org/abs/2504.05657
- In-the-Wild paper / dataset (Müller et al. 2022): https://deepfake-total.com/in_the_wild
"""


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    write_notebook(NOTEBOOK_DIR / "eval_asvspoof_2019.ipynb", build_asv2019())
    write_notebook(NOTEBOOK_DIR / "eval_asvspoof_2021.ipynb", build_asv2021())
    write_notebook(NOTEBOOK_DIR / "eval_asvspoof_5.ipynb", build_asv5())
    write_notebook(NOTEBOOK_DIR / "eval_in_the_wild.ipynb", build_in_the_wild())
    (NOTEBOOK_DIR / "evaluation_notebooks_summary.md").write_text(SUMMARY_MD, encoding="utf-8")
    print("evaluation notebooks written")


if __name__ == "__main__":
    main()

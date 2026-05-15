import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "visualize_asvspoof5_hard_errors.ipynb"


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


cells = [
    md(
        r"""
# ASVspoof 5 hard-error visualization

This notebook is a qualitative/triage tool for ASVspoof 5 Track 1 hard errors. It prioritizes reconstructing per-model confident errors from `error_analysis.pkl`. If that artifact does not expose full per-utterance scores, it falls back to the exported `hard_errors.csv` and labels it honestly as the available export, which may be AASIST-only in older runs.

Audio visualization is disabled by default because ASVspoof 5 eval audio is stored in large Hugging Face tar shards (~8.45 GB per shard). Set `RUN_AUDIO_VIZ = True` only for a small selected subset.
"""
    ),
    code(
        r"""
import gc
import io
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "seaborn"])
    import matplotlib.pyplot as plt
    import seaborn as sns

try:
    import soundfile as sf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile"])
    import soundfile as sf

try:
    import librosa
    import librosa.display
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "librosa"])
    import librosa
    import librosa.display

try:
    from scipy.fftpack import dct
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.fftpack import dct

from IPython.display import Audio, display

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 140)

ARTIFACT_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/minhbhm/sdd-results/error_analysis/asvspoof5"),
    Path("/kaggle/input/sdd-results/error_analysis/asvspoof5"),
    Path("results/error_analysis/asvspoof5"),
]
RESULTS_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/minhbhm/sdd-results"),
    Path("/kaggle/input/sdd-results"),
    Path("results"),
]

OUTPUT_DIR = Path("/kaggle/working/asv5_hard_error_visualization")
PLOT_DIR = OUTPUT_DIR / "plots"
AUDIO_DIR = OUTPUT_DIR / "audio_snippets"
for path in (OUTPUT_DIR, PLOT_DIR, AUDIO_DIR):
    path.mkdir(parents=True, exist_ok=True)

RUN_AUDIO_VIZ = False
MAX_AUDIO_UTTS = 24
N_PER_GROUP = 5
TOP_K_PER_MODEL = 60

# Buộc đưa vào selection các confident FP đại diện cho hai họ kiến trúc.
# Mục đích: §3.4.4 cần evidence trực quan của lỗi false_positive trên modern
# attack — loại lỗi nguy hiểm về security. Top FP rate trên ASV5 eval verified
# từ error_analysis.pkl: A28 #1 cho cả XLS-R+AASIST (0.17%) và XLS-R+Nes2Net
# (3.01%); A17 #1 cho AASIST (11.96%) và top-2 cho AASIST-L, đồng thời cũng nằm
# top-2 của hai XLS-R model — cross-cutting attack.
#
# Force theo cặp (utt_id, model) — không phải chỉ utt_id — vì một audio có thể
# fail nhiều model, và caption figure phải khớp với model được claim trong
# report.md. Các target này có thể nằm ngoài top-TOP_K_PER_MODEL của hard_long,
# nên được inject trực tiếp từ score_df full (xem cell tiếp theo).
FORCED_INCLUDE_TARGETS = [
    ("E_0004782150", "XLS-R+Nes2Net"),  # A28 nocodec, FP rate #1 cho XLS-R+Nes2Net
    ("E_0009392843", "XLS-R+Nes2Net"),  # A28 nocodec, FP cho XLS-R+Nes2Net
    ("E_0002841318", "AASIST"),         # A17 nocodec, FP rate #1 cho AASIST
    ("E_0002067090", "AASIST"),         # A17 nocodec, FP cho AASIST; cũng FP score cao trên cả hai XLS-R
]
FORCED_INCLUDE_UTT_IDS = [uid for uid, _ in FORCED_INCLUDE_TARGETS]
ASV5_SPLIT = "eval"
ASV5_TRACK = "track_1"
ASV5_HF_REPO_ID = "jungjee/asvspoof5"
ASV5_CACHE_DIR = Path("/kaggle/working/asvspoof5_hf_cache")
ASV5_TAR_WORK_DIR = Path("/kaggle/working/asvspoof5_tar_shards")
DELETE_ASV5_TAR_AFTER_USE = True

MODELS = ["AASIST", "AASIST-L", "LFCC+LCNN", "XLS-R+Nes2Net", "XLS-R+AASIST"]
SCORE_COLS = {model: f"{model}_score" for model in MODELS}
SCORE_COLS["XLS-R+Nes2Net"] = "XLS-R+Nes2Net_score"
SCORE_COLS["XLS-R+AASIST"] = "XLS-R+AASIST_score"

CODEC_DESCRIPTIONS = {
    "nocodec": "C00 / no encoding-compression",
    "C01": "Opus 16 kHz",
    "C02": "AMR 16 kHz",
    "C03": "Speex 16 kHz",
    "C04": "Encodec 16 kHz, 1.5-24.0 kbit/s",
    "C05": "MP3 16 kHz, 45-256 kbit/s",
    "C06": "M4A 16 kHz, 16-128 kbit/s",
    "C07": "MP3 + Encodec 16 kHz, varied",
    "C08": "Opus 8 kHz",
    "C09": "AMR 8 kHz",
    "C10": "Speex 8 kHz",
    "C11": "Varied calling/PSTN-like pipelines 8 kHz",
}

def first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None

ARTIFACT_ROOT = first_existing(ARTIFACT_ROOT_CANDIDATES)
RESULTS_ROOT = first_existing(RESULTS_ROOT_CANDIDATES)
if ARTIFACT_ROOT is None:
    raise FileNotFoundError(f"Could not find ASVspoof 5 error-analysis artifacts in: {ARTIFACT_ROOT_CANDIDATES}")
print("ARTIFACT_ROOT:", ARTIFACT_ROOT)
print("RESULTS_ROOT:", RESULTS_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
"""
    ),
    md(
        r"""
## Load artifacts and check data quality
"""
    ),
    code(
        r"""
hard_errors_path = ARTIFACT_ROOT / "hard_errors.csv"
metrics_path = ARTIFACT_ROOT / "metrics.csv"
error_pkl_path = ARTIFACT_ROOT / "error_analysis.pkl"
results_pkl_candidates = []
if RESULTS_ROOT is not None:
    results_pkl_candidates.extend([
        RESULTS_ROOT / "asvspoof5" / "results.pkl",
        RESULTS_ROOT / "results" / "error_analysis" / "asvspoof5" / "error_analysis.pkl",
    ])

hard_raw = pd.read_csv(hard_errors_path)
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

print("hard_errors.csv:", hard_errors_path)
print("shape:", hard_raw.shape)
display(hard_raw.head())
print("columns:", list(hard_raw.columns))
print("\nMissingness:")
display(hard_raw.isna().mean().sort_values(ascending=False).to_frame("missing_fraction").head(20))

score_cols_present = [c for c in hard_raw.columns if c.endswith("_score")]
non_null_scores = hard_raw[score_cols_present].notna().sum().sort_values(ascending=False) if score_cols_present else pd.Series(dtype=int)
print("\nNon-null score counts:")
display(non_null_scores.to_frame("non_null"))

model_missing_rate = hard_raw["model"].isna().mean() if "model" in hard_raw.columns else 1.0
print(f"model missing rate: {model_missing_rate:.3f}")
if model_missing_rate > 0.5:
    print("WARNING: Most rows have missing model. This export may be an AASIST-only confident-error export, not a clean per-model hard-error table.")
"""
    ),
    md(
        r"""
## Reconstruct per-model confident errors when possible

The notebook searches `error_analysis.pkl` for a large DataFrame with `utt_id`, labels/metadata, and model score columns. If found, it reconstructs top confident false positives/false negatives per model. Otherwise, it falls back to `hard_errors.csv`.
"""
    ),
    code(
        r"""
def load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def iter_dataframes(obj, prefix="root"):
    if isinstance(obj, pd.DataFrame):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_dataframes(value, f"{prefix}.{key}")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_dataframes(value, f"{prefix}[{idx}]")


def normalize_score_column_candidates(df: pd.DataFrame) -> dict[str, str]:
    cols = set(df.columns)
    mapping = {}
    aliases = {
        "AASIST": ["AASIST_score", "score_AASIST", "AASIST"],
        "AASIST-L": ["AASIST-L_score", "score_AASIST-L", "AASIST_L_score", "AASIST-L"],
        "LFCC+LCNN": ["LFCC+LCNN_score", "score_LFCC+LCNN", "LFCC_LCNN_score", "LFCC+LCNN"],
        "XLS-R+Nes2Net": ["XLS-R+Nes2Net_score", "score_XLS-R+Nes2Net", "XLSR_Nes2Net_score", "XLS-R+Nes2Net"],
        "XLS-R+AASIST": ["XLS-R+AASIST_score", "score_XLS-R+AASIST", "XLSR_AASIST_score", "XLS-R+AASIST"],
    }
    for model, names in aliases.items():
        for name in names:
            if name in cols and pd.api.types.is_numeric_dtype(df[name]):
                mapping[model] = name
                break
    return mapping


def find_best_score_dataframe(obj):
    candidates = []
    for name, df in iter_dataframes(obj):
        if "utt_id" not in df.columns:
            continue
        score_map = normalize_score_column_candidates(df)
        label_col = "label" if "label" in df.columns else ("y_true" if "y_true" in df.columns else None)
        if label_col is None or not score_map:
            continue
        candidates.append((len(score_map), len(df), name, df, score_map, label_col))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0]


def reconstruct_hard_errors_from_df(df: pd.DataFrame, score_map: dict[str, str], label_col: str, top_k: int = TOP_K_PER_MODEL) -> pd.DataFrame:
    meta_cols = [c for c in ["utt_id", "label", "speaker", "attack", "attack_tag", "codec", "condition", "gender", "codec_q"] if c in df.columns]
    rows = []
    work = df.copy()
    if "label" not in work.columns:
        work["label"] = work[label_col].astype(int)
        if "label" not in meta_cols:
            meta_cols = ["label"] + meta_cols
    for model, score_col in score_map.items():
        sub = work[meta_cols + [score_col]].copy()
        sub = sub.rename(columns={score_col: "score"})
        sub = sub[sub["score"].notna()].copy()
        if sub.empty:
            continue
        sub["model"] = model
        sub["error_type"] = np.where(
            (sub["label"].astype(int) == 0) & (sub["score"] >= 0.5),
            "false_positive",
            np.where((sub["label"].astype(int) == 1) & (sub["score"] < 0.5), "false_negative", "correct"),
        )
        sub = sub[sub["error_type"] != "correct"].copy()
        if sub.empty:
            continue
        sub["confidence"] = np.where(sub["error_type"].eq("false_positive"), sub["score"], 1.0 - sub["score"])
        sub = sub.sort_values("confidence", ascending=False).head(top_k)
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fallback_hard_errors_long(hard_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in ["utt_id", "label", "speaker", "attack", "attack_tag", "codec", "condition", "error_type", "confidence", "failed_model_count", "model"] if c in hard_df.columns]
    if "model" in hard_df.columns and hard_df["model"].notna().any():
        out = hard_df[base_cols].copy()
        if "score" not in out.columns:
            out["score"] = np.nan
        return out
    # Older export: rows are effectively AASIST confident errors with AASIST_score populated.
    out = hard_df[base_cols + (["AASIST_score"] if "AASIST_score" in hard_df.columns else [])].copy()
    out["model"] = out.get("model", pd.Series(index=out.index, dtype=object)).fillna("AASIST")
    if "AASIST_score" in out.columns:
        out["score"] = out["AASIST_score"]
        out = out.drop(columns=["AASIST_score"])
    else:
        out["score"] = np.nan
    return out


error_obj = load_pickle(error_pkl_path)
source_mode = "hard_errors_csv_fallback"
hard_long = pd.DataFrame()
score_df_info = None

if error_obj is not None:
    score_df_info = find_best_score_dataframe(error_obj)
    if score_df_info is not None:
        n_models, n_rows, name, score_df, score_map, label_col = score_df_info
        print(f"Found score DataFrame in error_analysis.pkl: {name}, rows={n_rows:,}, models={list(score_map)}")
        hard_long = reconstruct_hard_errors_from_df(score_df, score_map, label_col)
        if not hard_long.empty:
            source_mode = "reconstructed_from_error_analysis_pkl"

if hard_long.empty:
    print("WARNING: Could not reconstruct per-model hard errors from error_analysis.pkl. Falling back to hard_errors.csv.")
    hard_long = fallback_hard_errors_long(hard_raw)


def append_forced_targets(hard_long_df: pd.DataFrame, score_df_info_tuple, forced_targets) -> pd.DataFrame:
    # Inject forced (utt_id, model) pairs trực tiếp từ score_df full.
    # Các target này có thể nằm ngoài top-TOP_K_PER_MODEL của reconstruct_hard_errors_from_df,
    # nên không thể lọc bằng utt_id trên hard_long. Hàm tra cứu trực tiếp score_df, build
    # row với đúng score/error_type/confidence cho model được chỉ định, rồi prepend vào hard_long.
    if score_df_info_tuple is None or not forced_targets:
        return hard_long_df
    _, _, _, score_df_full, score_map_dict, label_col_name = score_df_info_tuple
    meta_cols = [c for c in ["utt_id", "label", "speaker", "attack", "attack_tag", "codec", "condition", "gender", "codec_q"] if c in score_df_full.columns]
    rows = []
    for utt_id, model in forced_targets:
        if model not in score_map_dict:
            print(f"WARNING forced ({utt_id}, {model}): model không có trong score_map ({list(score_map_dict)})")
            continue
        score_col = score_map_dict[model]
        match = score_df_full[score_df_full["utt_id"].astype(str) == utt_id]
        if match.empty:
            print(f"WARNING forced ({utt_id}, {model}): utt_id không có trong score_df")
            continue
        if score_col not in match.columns or pd.isna(match.iloc[0][score_col]):
            print(f"WARNING forced ({utt_id}, {model}): không có score cho model này")
            continue
        row = {c: match.iloc[0][c] for c in meta_cols}
        score_val = float(match.iloc[0][score_col])
        label_val = int(row.get("label", match.iloc[0][label_col_name]))
        if label_val == 0 and score_val >= 0.5:
            error_type = "false_positive"
            confidence = score_val
        elif label_val == 1 and score_val < 0.5:
            error_type = "false_negative"
            confidence = 1.0 - score_val
        else:
            print(f"WARNING forced ({utt_id}, {model}): không phải error (label={label_val}, score={score_val:.4f})")
            continue
        row["model"] = model
        row["score"] = score_val
        row["error_type"] = error_type
        row["confidence"] = confidence
        rows.append(row)
    if not rows:
        return hard_long_df
    forced_df = pd.DataFrame(rows)
    print(f"Forced-injected {len(forced_df)} rows từ score_df full: {[(r['utt_id'], r['model'], r['attack'], r['codec']) for r in rows]}")
    return pd.concat([forced_df, hard_long_df], ignore_index=True)


hard_long = append_forced_targets(hard_long, score_df_info, FORCED_INCLUDE_TARGETS)

print("source_mode:", source_mode)
print("hard_long shape:", hard_long.shape)
display(hard_long.head())
hard_long.to_csv(OUTPUT_DIR / "hard_errors_long.csv", index=False)
print("saved:", OUTPUT_DIR / "hard_errors_long.csv")
"""
    ),
    md(
        r"""
## CSV-only breakdowns
"""
    ),
    code(
        r"""
def save_current_fig(name: str):
    path = PLOT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print("saved:", path)


def bar_count(df, col, title, top_n=20, filename=None):
    if col not in df.columns:
        print(f"skip {title}: missing {col}")
        return
    counts = df[col].fillna("missing").astype(str).value_counts().head(top_n)
    plt.figure(figsize=(10, max(3, 0.35 * len(counts))))
    sns.barplot(x=counts.values, y=counts.index, orient="h")
    plt.xlabel("count")
    plt.ylabel(col)
    plt.title(title)
    if filename:
        save_current_fig(filename)
    plt.show()


bar_count(hard_long, "model", "Hard errors by model", filename="hard_errors_by_model.png")
bar_count(hard_long, "error_type", "Hard errors by error type", filename="hard_errors_by_error_type.png")
bar_count(hard_long, "attack", "Top hard-error attacks", filename="hard_errors_by_attack.png")
bar_count(hard_long, "attack_tag", "Hard errors by attack tag", filename="hard_errors_by_attack_tag.png")
bar_count(hard_long, "codec", "Hard errors by codec", filename="hard_errors_by_codec.png")
bar_count(hard_long, "speaker", "Top hard-error speakers", top_n=25, filename="hard_errors_by_speaker.png")

if "confidence" in hard_long.columns:
    plt.figure(figsize=(9, 4))
    sns.histplot(pd.to_numeric(hard_long["confidence"], errors="coerce").dropna(), bins=40)
    for x in [0.9, 0.99, 0.999, 0.9999]:
        plt.axvline(x, color="red", linestyle="--", alpha=0.35)
    plt.xlabel("confidence")
    plt.title("Hard-error confidence distribution")
    save_current_fig("hard_error_confidence_hist.png")
    plt.show()
"""
    ),
    code(
        r"""
def heatmap_counts(df, row, col, title, filename):
    if row not in df.columns or col not in df.columns:
        print(f"skip {title}: missing {row} or {col}")
        return
    tab = pd.crosstab(df[row].fillna("missing"), df[col].fillna("missing"))
    if tab.empty:
        return
    plt.figure(figsize=(max(8, 0.7 * tab.shape[1]), max(4, 0.45 * tab.shape[0])))
    sns.heatmap(tab, annot=True, fmt="d", cmap="mako")
    plt.title(title)
    save_current_fig(filename)
    plt.show()


heatmap_counts(hard_long, "attack", "codec", "Hard-error count: attack x codec", "hard_errors_attack_codec_heatmap.png")
heatmap_counts(hard_long, "model", "codec", "Hard-error count: model x codec", "hard_errors_model_codec_heatmap.png")
heatmap_counts(hard_long, "model", "attack", "Hard-error count: model x attack", "hard_errors_model_attack_heatmap.png")

if "failed_model_count" in hard_long.columns and hard_long["failed_model_count"].notna().any():
    bar_count(hard_long, "failed_model_count", "Failed model count distribution", filename="failed_model_count_distribution.png")
"""
    ),
    md(
        r"""
## Score context and cross-model behavior

These cells run only when reconstructed/joined score columns are available. They answer whether a confident error for one model is also difficult for other models.
"""
    ),
    code(
        r"""
score_context = None
if score_df_info is not None:
    _, _, name, score_df, score_map, label_col = score_df_info
    keep_cols = [c for c in ["utt_id", "label", "speaker", "attack", "attack_tag", "codec", "condition"] if c in score_df.columns]
    score_context = score_df[keep_cols + list(score_map.values())].copy()
    rename = {v: f"{k}_score" for k, v in score_map.items()}
    score_context = score_context.rename(columns=rename)
    hard_ids = hard_long["utt_id"].astype(str).unique()
    hard_score_context = score_context[score_context["utt_id"].astype(str).isin(hard_ids)].copy()
    print("hard_score_context:", hard_score_context.shape)
    display(hard_score_context.head())
else:
    hard_score_context = pd.DataFrame()
    print("No full score context available.")

score_cols = [c for c in hard_score_context.columns if c.endswith("_score")]
if score_cols:
    melted = hard_score_context.melt(
        id_vars=[c for c in ["utt_id", "label", "attack", "attack_tag", "codec"] if c in hard_score_context.columns],
        value_vars=score_cols,
        var_name="model",
        value_name="score",
    )
    melted["model"] = melted["model"].str.replace("_score", "", regex=False)
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=melted, x="model", y="score")
    sns.stripplot(data=melted, x="model", y="score", color="black", alpha=0.25, size=2)
    plt.xticks(rotation=20, ha="right")
    plt.title("Cross-model scores on selected hard-error utterances")
    save_current_fig("cross_model_scores_on_hard_errors.png")
    plt.show()

    preview_cols = ["utt_id", "label", "attack", "attack_tag", "codec"] + score_cols
    display(hard_score_context[preview_cols].head(30))
"""
    ),
    code(
        r"""
if score_context is not None:
    score_cols = [c for c in score_context.columns if c.endswith("_score")]
    fig, axes = plt.subplots(len(score_cols), 1, figsize=(10, 2.4 * len(score_cols)), sharex=True)
    if len(score_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, score_cols):
        sns.histplot(score_context[col].dropna(), bins=80, ax=ax, color="steelblue", alpha=0.6)
        if not hard_score_context.empty and col in hard_score_context.columns:
            y_top = ax.get_ylim()[1]
            hard_vals = hard_score_context[col].dropna()
            ax.scatter(hard_vals, np.full(len(hard_vals), y_top * 0.95), marker="|", color="red", s=80, label="hard-error utt")
        ax.set_title(col)
        ax.set_ylabel("count")
    axes[-1].set_xlabel("bonafide score")
    save_current_fig("hard_error_rug_on_score_distributions.png")
    plt.show()
"""
    ),
    md(
        r"""
## Select utterances for optional audio visualization
"""
    ),
    code(
        r"""
def select_audio_targets(df: pd.DataFrame, max_utts: int = MAX_AUDIO_UTTS) -> pd.DataFrame:
    work = df.copy()
    if "confidence" in work.columns:
        work["confidence_num"] = pd.to_numeric(work["confidence"], errors="coerce")
    else:
        work["confidence_num"] = np.nan
    # Force theo cặp (utt_id, model) — không phải chỉ utt_id — để tránh trường hợp
    # cùng utt_id fail nhiều model, drop_duplicates(utt_id) chọn nhầm model có
    # confidence cao hơn nhưng không phải target claim trong report.
    forced_parts = []
    for forced_utt_id, forced_model in FORCED_INCLUDE_TARGETS:
        sub = work[
            work["utt_id"].astype(str).eq(forced_utt_id)
            & work["model"].astype(str).eq(forced_model)
        ].copy()
        if sub.empty:
            print(f"WARNING: forced target missing trong hard_long: ({forced_utt_id}, {forced_model})")
            continue
        forced_parts.append(sub.sort_values("confidence_num", ascending=False).head(1))
    forced_rows = pd.concat(forced_parts, ignore_index=True) if forced_parts else pd.DataFrame()
    parts = []
    for codec in ["C04", "C07", "nocodec", "C05", "C06", "C11"]:
        sub = work[work.get("codec", pd.Series(index=work.index, dtype=object)).eq(codec)].sort_values("confidence_num", ascending=False)
        if not sub.empty:
            parts.append(sub.head(N_PER_GROUP))
    for err in ["false_positive", "false_negative"]:
        sub = work[work.get("error_type", pd.Series(index=work.index, dtype=object)).eq(err)].sort_values("confidence_num", ascending=False)
        if not sub.empty:
            parts.append(sub.head(N_PER_GROUP))
    if parts:
        selected = pd.concat(parts, ignore_index=True)
    else:
        selected = work.sort_values("confidence_num", ascending=False)
    # Forced rows luôn được giữ và đặt lên đầu để không bị head(max_utts) cắt mất.
    if not forced_rows.empty:
        selected = pd.concat([forced_rows, selected], ignore_index=True)
    selected = selected.drop_duplicates("utt_id").head(max_utts).copy()
    selected_pairs = set(zip(selected["utt_id"].astype(str), selected["model"].astype(str)))
    missing_forced_pairs = [pair for pair in FORCED_INCLUDE_TARGETS if pair not in selected_pairs]
    if missing_forced_pairs:
        print(f"WARNING: forced (utt_id, model) pairs không có trong final selection: {missing_forced_pairs}")
    return selected


selected = select_audio_targets(hard_long)
selected.to_csv(OUTPUT_DIR / "selected_hard_errors.csv", index=False)
print("selected for optional audio:", selected.shape)
display(selected[["utt_id", "model", "label", "attack", "attack_tag", "codec", "error_type", "confidence"]].head(MAX_AUDIO_UTTS))
print("saved:", OUTPUT_DIR / "selected_hard_errors.csv")
"""
    ),
    md(
        r"""
## Optional audio loading from ASVspoof 5 Hugging Face tar shards

This is disabled by default. When enabled, the notebook downloads one tar shard at a time, extracts only selected utterances, then deletes the shard. Each eval shard is large (~8.45 GB), so keep `MAX_AUDIO_UTTS` small.
"""
    ),
    code(
        r"""
def ensure_hf_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    return hf_hub_download


def hf_download_asv5_file(filename: str) -> Path:
    ASV5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download = ensure_hf_hub()
    return Path(hf_hub_download(repo_id=ASV5_HF_REPO_ID, repo_type="dataset", filename=filename, cache_dir=str(ASV5_CACHE_DIR)))


def hf_download_asv5_tar_shard(filename: str) -> Path:
    ASV5_TAR_WORK_DIR.mkdir(parents=True, exist_ok=True)
    hf_hub_download = ensure_hf_hub()
    return Path(hf_hub_download(repo_id=ASV5_HF_REPO_ID, repo_type="dataset", filename=filename, local_dir=str(ASV5_TAR_WORK_DIR)))


def remove_asv5_tar_shard(path: Path):
    if DELETE_ASV5_TAR_AFTER_USE and path.exists():
        path.unlink()
        print("deleted shard:", path)


def hf_tar_names_for_split(split: str) -> list[str]:
    if split == "train":
        return [f"flac_T_{suffix}.tar" for suffix in ["aa", "ab", "ac", "ad", "ae"]]
    if split == "dev":
        return [f"flac_D_{suffix}.tar" for suffix in ["aa", "ab", "ac"]]
    if split == "eval":
        return [f"flac_E_{suffix}.tar" for suffix in ["aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai", "aj"]]
    raise ValueError(split)


def protocol_member_candidates(split: str) -> list[str]:
    if split == "train":
        return ["ASVspoof5.train.tsv", "train.tsv"]
    return [f"ASVspoof5.{split}.{ASV5_TRACK}.tsv", f"{split}.{ASV5_TRACK}.tsv", f"{split}.track_1.tsv"]


def read_protocol_from_hf_tar(split: str) -> str:
    proto_tar = hf_download_asv5_file("ASVspoof5_protocols.tar")
    wanted = protocol_member_candidates(split)
    with tarfile.open(proto_tar, "r:*") as tar:
        for member in tar:
            if Path(member.name).name not in wanted:
                continue
            extracted = tar.extractfile(member)
            if extracted is not None:
                return extracted.read().decode("utf-8")
    raise FileNotFoundError(f"Could not find one of {wanted} in {proto_tar}")


def load_asv5_protocol_df(split: str = ASV5_SPLIT) -> pd.DataFrame:
    text = read_protocol_from_hf_tar(split)
    rows = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        speaker, utt_id, gender, codec, codec_q, codec_seed, attack_tag, attack_label, key, tmp = parts[:10]
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
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Parsed zero protocol rows for split={split}")
    return df


def attach_bonafide_references(selected_df: pd.DataFrame, protocol_df: pd.DataFrame) -> pd.DataFrame:
    # Attach one bonafide reference, preferring same speaker+codec and falling back to same codec.
    bonafide = protocol_df[protocol_df["label"].eq(1)].copy()
    out = selected_df.copy()
    refs = []
    ref_modes = []
    for _, row in out.iterrows():
        same = bonafide[
            bonafide["speaker"].astype(str).eq(str(row.get("speaker")))
            & bonafide["codec"].astype(str).eq(str(row.get("codec")))
        ]
        mode = "same_speaker_codec"
        if same.empty:
            same = bonafide[bonafide["codec"].astype(str).eq(str(row.get("codec")))]
            mode = "same_codec"
        if same.empty:
            same = bonafide
            mode = "any_bonafide"
        if same.empty:
            refs.append(None)
            ref_modes.append("none")
        else:
            ref = same.iloc[0]
            refs.append(ref["utt_id"])
            ref_modes.append(mode)
    out["bonafide_ref_utt_id"] = refs
    out["bonafide_ref_mode"] = ref_modes
    return out


def load_selected_audio_from_hf(selected_ids: set[str]) -> dict[str, bytes]:
    found = {}
    remaining = set(selected_ids)
    if not remaining:
        return found
    for filename in hf_tar_names_for_split(ASV5_SPLIT):
        if not remaining:
            break
        print(f"downloading/scanning {filename}; remaining={len(remaining)}")
        tar_path = hf_download_asv5_tar_shard(filename)
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    utt_id = Path(member.name).stem
                    if utt_id not in remaining:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    found[utt_id] = extracted.read()
                    remaining.remove(utt_id)
                    print("found", utt_id, f"({len(found)}/{len(selected_ids)})")
                    if not remaining:
                        break
        finally:
            remove_asv5_tar_shard(tar_path)
            gc.collect()
    if remaining:
        print("WARNING: missing selected audio ids:", sorted(remaining)[:20])
    return found


selected_audio = {}
protocol_df = None
if RUN_AUDIO_VIZ:
    protocol_df = load_asv5_protocol_df()
    selected = attach_bonafide_references(selected.head(MAX_AUDIO_UTTS), protocol_df)
    display(selected[["utt_id", "speaker", "codec", "bonafide_ref_utt_id", "bonafide_ref_mode"]])
    target_ids = set(selected["utt_id"].astype(str).head(MAX_AUDIO_UTTS))
    target_ids.update(selected["bonafide_ref_utt_id"].dropna().astype(str).tolist())
    print(f"RUN_AUDIO_VIZ=True; loading up to {len(target_ids)} utterances from HF tar shards.")
    selected_audio = load_selected_audio_from_hf(target_ids)
else:
    print("RUN_AUDIO_VIZ=False; skipping HF audio download.")
"""
    ),
    md(
        r"""
## Optional audio panels: LTAS, waveform, spectrograms, LFCC and descriptors
"""
    ),
    code(
        r"""
def read_audio_bytes_16k(audio_bytes: bytes):
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    return y.astype(np.float32), sr


def ltas_db(y, sr, n_fft=1024, hop_length=256):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mean_power = np.maximum(S.mean(axis=1), 1e-12)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, librosa.power_to_db(mean_power, ref=np.max)


def lfcc_features(y, sr, n_fft=1024, hop_length=256, n_filters=40, n_lfcc=40):
    # Lightweight LFCC-style representation using linearly spaced triangular filters.
    power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(fft_freqs)), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        up = (fft_freqs - left) / max(center - left, 1e-8)
        down = (right - fft_freqs) / max(right - center, 1e-8)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    energies = np.maximum(fb @ power, 1e-10)
    log_energies = np.log(energies)
    return dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc]


def plot_audio_panel(row: pd.Series, audio_bytes: bytes):
    y, sr = read_audio_bytes_16k(audio_bytes)
    ref_id = row.get("bonafide_ref_utt_id")
    ref_y = ref_sr = None
    if isinstance(ref_id, str) and ref_id in selected_audio:
        ref_y, ref_sr = read_audio_bytes_16k(selected_audio[ref_id])
    utt_id = row["utt_id"]
    wav_path = AUDIO_DIR / f"{utt_id}.wav"
    sf.write(wav_path, y, sr)

    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(6, 1, height_ratios=[1.0, 1.0, 1.5, 1.5, 1.2, 1.0])

    ax = fig.add_subplot(gs[0])
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(f"Waveform: {utt_id} | model={row.get('model')} | {row.get('error_type')} | codec={row.get('codec')} | attack={row.get('attack')}")

    ax = fig.add_subplot(gs[1])
    freqs, ltas = ltas_db(y, sr)
    ax.plot(freqs, ltas, label=f"hard error {utt_id}")
    if ref_y is not None:
        ref_freqs, ref_ltas = ltas_db(ref_y, ref_sr)
        ax.plot(ref_freqs, ref_ltas, label=f"bonafide ref {ref_id}", alpha=0.8)
    ax.set_xlim(0, sr / 2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_title("LTAS (long-term average spectrum)")
    ax.legend(loc="best")

    ax = fig.add_subplot(gs[2])
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="linear", ax=ax)
    ax.set_title("Linear STFT spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    ax = fig.add_subplot(gs[3])
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title("Log-mel spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    ax = fig.add_subplot(gs[4])
    lfcc = lfcc_features(y, sr)
    img = librosa.display.specshow(lfcc, sr=sr, hop_length=256, x_axis="time", ax=ax)
    ax.set_title("LFCC-style linear cepstral coefficients")
    fig.colorbar(img, ax=ax)

    ax = fig.add_subplot(gs[5])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=256, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=256)[0]
    times = librosa.times_like(cent, sr=sr, hop_length=256)
    ax.plot(times, cent, label="centroid Hz")
    ax.plot(times, roll, label="rolloff Hz", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(times, rms, color="tab:green", label="RMS", alpha=0.5)
    ax2.plot(times, zcr, color="tab:red", label="ZCR", alpha=0.5)
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=256)
        if f0 is not None and np.isfinite(f0).any():
            ax.plot(times[: len(f0)], f0, label="F0 Hz", color="tab:purple", alpha=0.65)
    except Exception as exc:
        print(f"pitch skipped for {utt_id}: {type(exc).__name__}: {exc}")
    ax.set_title("Spectral centroid / rolloff plus RMS / ZCR")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    out_path = PLOT_DIR / f"audio_panel_{utt_id}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print("saved:", out_path)
    display(Audio(y, rate=sr))
    return wav_path


if RUN_AUDIO_VIZ and selected_audio:
    for _, row in selected.head(MAX_AUDIO_UTTS).iterrows():
        utt_id = str(row["utt_id"])
        if utt_id not in selected_audio:
            continue
        plot_audio_panel(row, selected_audio[utt_id])
else:
    print("Audio panels skipped.")
"""
    ),
    md(
        r"""
## Export triage table
"""
    ),
    code(
        r"""
triage_cols = [c for c in [
    "utt_id", "model", "label", "speaker", "attack", "attack_tag", "codec", "condition",
    "score", "confidence", "error_type", "failed_model_count"
] if c in hard_long.columns]
triage = hard_long[triage_cols].copy()
if "codec" in triage.columns:
    triage["codec_description"] = triage["codec"].map(CODEC_DESCRIPTIONS).fillna(triage["codec"])
triage_path = OUTPUT_DIR / "hard_errors_triage.csv"
triage.to_csv(triage_path, index=False)
display(triage.head(50))
print("saved:", triage_path)
print("Notebook outputs:", OUTPUT_DIR)
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {NOTEBOOK_PATH}")

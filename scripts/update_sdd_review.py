from __future__ import annotations

import base64
import csv
import json
import re
from pathlib import Path


NOTEBOOK_PATH = Path("sdd-review.ipynb")
EXPORT_DIR = Path("notebook_exports") / "part6"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def get_text_output(cell: dict) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            text = output["text"]
            parts.append("".join(text) if isinstance(text, list) else text)
        elif "data" in output and "text/plain" in output["data"]:
            text = output["data"]["text/plain"]
            parts.append("".join(text) if isinstance(text, list) else text)
    return "".join(parts)


def get_stream_output(cell: dict) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            text = output["text"]
            parts.append("".join(text) if isinstance(text, list) else text)
    return "".join(parts)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_png_from_output(cell: dict, output_idx: int, path: Path) -> None:
    output = cell.get("outputs", [])[output_idx]
    b64 = output.get("data", {}).get("image/png")
    if not b64:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


def parse_table_from_stream(text: str) -> tuple[str, list[str], list[dict]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    header_idx = next(i for i, line in enumerate(lines) if "N_spoof" in line)
    header_tokens = lines[header_idx].split()
    group_col = header_tokens[0].lower()
    metric_cols = header_tokens[2:]

    rows: list[dict] = []
    for line in lines[header_idx + 2 :]:
        if set(line) == {"-"}:
            continue
        parts = line.rsplit(None, len(metric_cols) + 1)
        if len(parts) != len(metric_cols) + 2:
            continue
        row = {group_col: parts[0].strip(), "n_spoof": int(parts[1].replace(",", ""))}
        for col, val in zip(metric_cols, parts[2:]):
            row[col] = float(val.rstrip("%"))
        rows.append(row)

    return group_col, metric_cols, rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_failure_overlap(text: str) -> list[dict]:
    rows: list[dict] = []
    blocks = [block.strip() for block in text.split("\n\n") if "vs" in block and "Overlap ratio" in block]
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        pair = lines[0].rstrip(":")
        metrics: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metrics[key.strip()] = value.strip()

        left_model, right_model = [part.strip() for part in pair.split(" vs ", 1)]
        overlap_str = metrics.get("Overlap ratio", "nan%").split()[0].rstrip("%")
        rows.append(
            {
                "pair": pair,
                "model_1": left_model,
                "model_2": right_model,
                "errors_model_1": int(metrics[f"{left_model} errors"].replace(",", "")),
                "errors_model_2": int(metrics[f"{right_model} errors"].replace(",", "")),
                "both_wrong": int(metrics["Both wrong"].replace(",", "")),
                "only_model_1": int(metrics[f"Only {left_model}"].replace(",", "")),
                "only_model_2": int(metrics[f"Only {right_model}"].replace(",", "")),
                "overlap_ratio_percent": float(overlap_str),
            }
        )
    return rows


def export_part6_outputs(nb: dict) -> list[Path]:
    exported: list[Path] = []
    mapping = {
        39: "6_0_loaded_results.txt",
        40: "6_1_protocol_summary.txt",
        41: "6_2_score_attachment_summary.txt",
        45: "6_6_failure_overlap.txt",
    }

    for cell_idx, filename in mapping.items():
        path = EXPORT_DIR / filename
        save_text(path, get_stream_output(nb["cells"][cell_idx]))
        exported.append(path)

    for cell_idx, stem in [(42, "6_3_eer_by_attack_2019"), (43, "6_4_eer_by_codec_2021"), (44, "6_5_eer_by_vocoder_2021")]:
        cell = nb["cells"][cell_idx]
        text = get_stream_output(cell)
        txt_path = EXPORT_DIR / f"{stem}.txt"
        csv_path = EXPORT_DIR / f"{stem}.csv"
        png_path = EXPORT_DIR / f"{stem}.png"
        save_text(txt_path, text)
        group_col, metric_cols, rows = parse_table_from_stream(text)
        write_csv(csv_path, rows, [group_col, "n_spoof", *metric_cols])
        save_png_from_output(cell, 1, png_path)
        exported.extend([txt_path, csv_path, png_path])

    failure_rows = parse_failure_overlap(get_stream_output(nb["cells"][45]))
    failure_csv = EXPORT_DIR / "6_6_failure_overlap.csv"
    write_csv(
        failure_csv,
        failure_rows,
        [
            "pair",
            "model_1",
            "model_2",
            "errors_model_1",
            "errors_model_2",
            "both_wrong",
            "only_model_1",
            "only_model_2",
            "overlap_ratio_percent",
        ],
    )
    exported.append(failure_csv)
    save_png_from_output(nb["cells"][45], 0, EXPORT_DIR / "6_6_score_correlation.png")
    exported.append(EXPORT_DIR / "6_6_score_correlation.png")

    report_lines = ["# Exported Part 6 Outputs", ""]
    for path in sorted(exported):
        report_lines.append(f"- {path.as_posix()}")
    report_path = EXPORT_DIR / "README.md"
    save_text(report_path, "\n".join(report_lines) + "\n")
    exported.append(report_path)
    return exported


PART7_MARKDOWN = """---
## Part 7: ASVspoof 5 Evaluation & Comparison to ASVspoof 2019/2021

Evaluate on ASVspoof 5 (2024) to test cross-dataset robustness under newer attacks, codec diversity, and more realistic conditions.

**Default split:** `dev` for manageable runtime on Kaggle. Change `ASV5_CONFIG['split']` to `'eval'` only if the full audio is mounted and long runtime is acceptable.  
**Goal:** compare each model not only by overall EER, but also by attack type, codec, and hardest-case groups across ASVspoof 2019, 2021, and 5.
"""


PART7_CELL_70 = r"""# ═══════════════════════════════════════════
# 7.0  Locate ASVspoof 5 and choose evaluation setup
# ═══════════════════════════════════════════
!pip install -q datasets soundfile

import os
import glob
from pathlib import Path

ASV5_CONFIG = {
    'split': 'dev',          # recommended first run on Kaggle
    'track': 'track_1',
    'max_items': None,       # set e.g. 50000 for a quick smoke test
    'streaming': False,      # use True only if HF download is too large
}

ASV5_RESULTS_DIR = f'{RESULTS_DIR}/asvspoof5'
os.makedirs(ASV5_RESULTS_DIR, exist_ok=True)

ASV5_PROTOCOL = None
ASV5_AUDIO_DIR = None
ASV5_SOURCE = None

protocol_patterns = [
    f"/kaggle/input/**/*{ASV5_CONFIG['split']}*{ASV5_CONFIG['track']}*.tsv",
    f"/kaggle/input/**/*{ASV5_CONFIG['track']}*{ASV5_CONFIG['split']}*.tsv",
    f"/kaggle/input/**/*{ASV5_CONFIG['split']}*.tsv",
]
audio_patterns = [
    f"/kaggle/input/**/*flac*{ASV5_CONFIG['split']}*",
    f"/kaggle/input/**/*{ASV5_CONFIG['split']}*flac*",
]

for pattern in protocol_patterns:
    candidates = sorted(glob.glob(pattern, recursive=True))
    if candidates:
        ASV5_PROTOCOL = candidates[0]
        break

for pattern in audio_patterns:
    candidates = [p for p in sorted(glob.glob(pattern, recursive=True)) if os.path.isdir(p)]
    if candidates:
        ASV5_AUDIO_DIR = candidates[0]
        break

if ASV5_PROTOCOL and ASV5_AUDIO_DIR:
    ASV5_SOURCE = 'local'
else:
    ASV5_SOURCE = 'huggingface'

print('ASVspoof 5 setup')
print('=' * 60)
print(f"  Split        : {ASV5_CONFIG['split']}")
print(f"  Track        : {ASV5_CONFIG['track']}")
print(f"  Source       : {ASV5_SOURCE}")
print(f"  Protocol     : {ASV5_PROTOCOL}")
print(f"  Audio dir    : {ASV5_AUDIO_DIR}")
print(f"  Max items    : {ASV5_CONFIG['max_items']}")
print(f"  Results dir  : {ASV5_RESULTS_DIR}")

if ASV5_CONFIG['split'] == 'eval':
    print('\n⚠️  eval split is large. Use dev first unless full run is intentional.')
"""


PART7_CELL_71 = r"""# ═══════════════════════════════════════════
# 7.1  Build ASVspoof 5 record list
# ═══════════════════════════════════════════
import pandas as pd
from itertools import islice
from datasets import load_dataset

def normalize_asv5_label(value):
    s = str(value).strip().lower()
    return 1 if 'bona' in s else 0

def normalize_asv5_group(value, default='unknown'):
    s = str(value).strip()
    return default if s in ('', '-', 'None', 'nan') else s

def find_first_key(sample, keys, fallback=None):
    for key in keys:
        if key in sample:
            return key
    return fallback

def load_asv5_local_records(protocol_path, audio_dir):
    df = pd.read_csv(protocol_path, sep='\t')
    if 'utt_id' not in df.columns or 'label' not in df.columns:
        df = pd.read_csv(protocol_path, sep='\t', header=None)
        rename_map = {
            0: 'speaker',
            1: 'utt_id',
            5: 'codec',
            6: 'attack',
            7: 'label',
        }
        df = df.rename(columns=rename_map)

    df['attack'] = df.get('attack', '-').map(lambda x: normalize_asv5_group(x, 'bonafide'))
    df['codec'] = df.get('codec', 'nocodec').map(lambda x: normalize_asv5_group(x, 'nocodec'))
    df['label'] = df['label'].map(normalize_asv5_label)

    audio_index = {}
    for path in Path(audio_dir).rglob('*.flac'):
        audio_index[path.stem] = str(path)

    df['audio_ref'] = df['utt_id'].map(audio_index)
    df = df[df['audio_ref'].notna()].reset_index(drop=True)

    if ASV5_CONFIG['max_items'] is not None:
        df = df.iloc[:ASV5_CONFIG['max_items']].copy()

    return [
        {
            'utt_id': row['utt_id'],
            'attack': row['attack'],
            'codec': row['codec'],
            'label': int(row['label']),
            'audio_ref': row['audio_ref'],
        }
        for _, row in df.iterrows()
    ]

def load_asv5_hf_records():
    split_candidates = [
        ASV5_CONFIG['split'],
        f"{ASV5_CONFIG['split']}.{ASV5_CONFIG['track']}",
        f"{ASV5_CONFIG['track']}_{ASV5_CONFIG['split']}",
    ]

    dataset = None
    last_error = None
    for split_name in split_candidates:
        try:
            dataset = load_dataset(
                'jungjee/asvspoof5',
                split=split_name,
                streaming=ASV5_CONFIG['streaming'],
            )
            print(f'Loaded HuggingFace split: {split_name}')
            break
        except Exception as exc:
            last_error = exc

    if dataset is None:
        raise RuntimeError(f'Cannot load ASVspoof 5 from HuggingFace: {last_error}')

    records = []
    iterator = dataset if ASV5_CONFIG['max_items'] is None else islice(dataset, ASV5_CONFIG['max_items'])

    for sample in iterator:
        utt_key = find_first_key(sample, ['utt_id', 'id', 'file', 'filename'], 'utt_id')
        attack_key = find_first_key(sample, ['attack', 'attack_id', 'spoofing_attack'], None)
        codec_key = find_first_key(sample, ['codec', 'codec_id', 'compression'], None)
        audio_key = find_first_key(sample, ['audio', 'path', 'speech'], None)
        label_key = find_first_key(sample, ['label', 'bonafide', 'class', 'target'], 'label')

        if audio_key is None:
            raise RuntimeError('Cannot infer ASVspoof 5 audio column from HuggingFace sample.')

        records.append(
            {
                'utt_id': str(sample.get(utt_key, f"sample_{len(records)}")),
                'attack': normalize_asv5_group(sample.get(attack_key, 'unknown'), 'unknown'),
                'codec': normalize_asv5_group(sample.get(codec_key, 'nocodec'), 'nocodec'),
                'label': normalize_asv5_label(sample.get(label_key, 0)),
                'audio_ref': sample[audio_key],
            }
        )

    return records

if ASV5_SOURCE == 'local':
    ASV5_RECORDS = load_asv5_local_records(ASV5_PROTOCOL, ASV5_AUDIO_DIR)
else:
    ASV5_RECORDS = load_asv5_hf_records()

print(f'ASVspoof 5 records ready: {len(ASV5_RECORDS):,}')
if ASV5_RECORDS:
    preview = pd.DataFrame(ASV5_RECORDS[:5]).drop(columns=['audio_ref'], errors='ignore')
    display(preview)

    attack_counts = pd.Series([r['attack'] for r in ASV5_RECORDS]).value_counts().head(10)
    codec_counts = pd.Series([r['codec'] for r in ASV5_RECORDS]).value_counts()
    print('\nTop attack groups:')
    print(attack_counts.to_string())
    print('\nCodec distribution:')
    print(codec_counts.to_string())
"""


PART7_CELL_72 = r"""# ═══════════════════════════════════════════
# 7.2  Evaluate models on ASVspoof 5
# ═══════════════════════════════════════════
import sys
import json
import importlib

def load_audio_16k(audio_ref):
    if isinstance(audio_ref, str):
        wav, sr = torchaudio.load(audio_ref)
    elif isinstance(audio_ref, dict):
        if 'array' not in audio_ref or 'sampling_rate' not in audio_ref:
            raise ValueError('Unsupported audio dict format.')
        wav = torch.from_numpy(audio_ref['array']).float().unsqueeze(0)
        sr = audio_ref['sampling_rate']
    else:
        raise TypeError(f'Unsupported audio reference type: {type(audio_ref)}')

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav

def predict_aasist_wav(model, wav, device='cuda', cut=64600):
    if wav.shape[1] < cut:
        wav = torch.nn.functional.pad(wav, (0, cut - wav.shape[1]))
    else:
        wav = wav[:, :cut]
    with torch.no_grad():
        _, out = model(wav.to(device))
    scores = out.softmax(dim=1).cpu().numpy()[0]
    return float(scores[1])  # bonafide score

def predict_aasist3_wav(model, wav, device='cuda', cut=64600):
    if wav.shape[1] < cut:
        wav = torch.nn.functional.pad(wav, (0, cut - wav.shape[1]))
    else:
        wav = wav[:, :cut]
    with torch.no_grad():
        out = model(wav.to(device))
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return float(probs[0])  # class 0 = bonafide for this checkpoint

def predict_lcnn_wav(model, wav, device='cuda', max_frames=400):
    feat = lfcc_transform(wav)
    T = feat.shape[2]
    if T < max_frames:
        feat = torch.nn.functional.pad(feat, (0, max_frames - T))
    else:
        feat = feat[:, :, :max_frames]
    with torch.no_grad():
        logits = model(feat.unsqueeze(0).to(device))
        score = torch.softmax(logits, dim=1)[0, 1].item()
    return score

def evaluate_asv5_model(model_name, model, predict_fn):
    scores, labels, meta = [], [], []
    errors = 0
    t0 = time.time()

    for record in tqdm(ASV5_RECORDS, desc=f'{model_name} → ASVspoof5'):
        try:
            wav = load_audio_16k(record['audio_ref'])
            score = predict_fn(model, wav, device)
            scores.append(score)
            labels.append(record['label'])
            meta.append({
                'utt_id': record['utt_id'],
                'attack': record['attack'],
                'codec': record['codec'],
            })
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f'  Error on {record.get("utt_id", "?")}: {type(exc).__name__}: {exc}')

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    eer = compute_eer(scores_arr, labels_arr) if len(scores_arr) else float('nan')

    return {
        'eer': eer,
        'scores': scores_arr,
        'labels': labels_arr,
        'meta': meta,
        'n_errors': errors,
        'elapsed_sec': time.time() - t0,
        'split': ASV5_CONFIG['split'],
    }

RESULTS_ASV5 = {}

if not Path('/kaggle/working/aasist').exists():
    !git clone https://github.com/clovaai/aasist.git /kaggle/working/aasist

sys.path.insert(0, '/kaggle/working/aasist')
os.chdir('/kaggle/working/aasist')

with open('config/AASIST.conf') as f:
    cfg = json.load(f)
mod = importlib.import_module(f'models.{cfg["model_config"]["architecture"]}')
aasist_model = mod.Model(cfg['model_config']).to(device)
aasist_model.load_state_dict(torch.load('models/weights/AASIST.pth', map_location=device))
aasist_model.eval()
RESULTS_ASV5['AASIST'] = evaluate_asv5_model('AASIST', aasist_model, predict_aasist_wav)
print(f"🎯 AASIST on ASVspoof5 {ASV5_CONFIG['split']}: EER = {RESULTS_ASV5['AASIST']['eer']:.2f}%")
del aasist_model
torch.cuda.empty_cache()

with open('config/AASIST-L.conf') as f:
    cfg_l = json.load(f)
mod_l = importlib.import_module(f'models.{cfg_l["model_config"]["architecture"]}')
aasist_l_model = mod_l.Model(cfg_l['model_config']).to(device)
aasist_l_model.load_state_dict(torch.load('models/weights/AASIST-L.pth', map_location=device))
aasist_l_model.eval()
RESULTS_ASV5['AASIST-L'] = evaluate_asv5_model('AASIST-L', aasist_l_model, predict_aasist_wav)
print(f"🎯 AASIST-L on ASVspoof5 {ASV5_CONFIG['split']}: EER = {RESULTS_ASV5['AASIST-L']['eer']:.2f}%")
del aasist_l_model
torch.cuda.empty_cache()

if Path('/kaggle/working/lfcc_lcnn.pth').exists():
    lcnn = SimpleLCNN().to(device)
    lcnn.load_state_dict(torch.load('/kaggle/working/lfcc_lcnn.pth', map_location=device))
    lcnn.eval()
    RESULTS_ASV5['LFCC+LCNN'] = evaluate_asv5_model('LFCC+LCNN', lcnn, predict_lcnn_wav)
    print(f"🎯 LFCC+LCNN on ASVspoof5 {ASV5_CONFIG['split']}: EER = {RESULTS_ASV5['LFCC+LCNN']['eer']:.2f}%")
    del lcnn
    torch.cuda.empty_cache()
else:
    print('⚠️  /kaggle/working/lfcc_lcnn.pth not found → skip LFCC+LCNN.')

try:
    if not Path('/kaggle/working/AASIST3').exists():
        !git clone https://github.com/mtuciru/AASIST3.git /kaggle/working/AASIST3
        !pip install -q transformers huggingface_hub

    sys.path.insert(0, '/kaggle/working/AASIST3')
    os.chdir('/kaggle/working/AASIST3')
    from model import aasist3

    aasist3_model = aasist3.from_pretrained('MTUCI/AASIST3').eval().to(device)
    RESULTS_ASV5['AASIST3'] = evaluate_asv5_model('AASIST3', aasist3_model, predict_aasist3_wav)
    print(f"🎯 AASIST3 on ASVspoof5 {ASV5_CONFIG['split']}: EER = {RESULTS_ASV5['AASIST3']['eer']:.2f}%")
    del aasist3_model
    torch.cuda.empty_cache()
except Exception as exc:
    print(f'⚠️  AASIST3 skipped: {type(exc).__name__}: {exc}')

os.chdir('/kaggle/working')

print('\nSummary:')
for name, data in sorted(RESULTS_ASV5.items(), key=lambda x: x[1]['eer']):
    print(f"  {name:15s}  EER = {data['eer']:.2f}%  (N={len(data['scores']):,}, errors={data['n_errors']})")
"""


PART7_CELL_73 = r"""# ═══════════════════════════════════════════
# 7.3  ASVspoof 5 grouped EER by attack and codec
# ═══════════════════════════════════════════
import matplotlib.pyplot as plt

def grouped_eer_from_result(result, group_col):
    df = pd.DataFrame(result['meta']).copy()
    df['score'] = result['scores']
    df['label'] = result['labels']

    bonafide = df[df['label'] == 1]
    rows = []
    for group_name, group_df in df[df['label'] == 0].groupby(group_col):
        subset = pd.concat([bonafide, group_df], ignore_index=True)
        rows.append({
            group_col: group_name,
            'n_spoof': len(group_df),
            'eer': compute_eer(subset['score'].values, subset['label'].values),
        })
    return pd.DataFrame(rows).sort_values(group_col)

ASV5_GROUP_TABLES = {}

for group_col, title in [('attack', 'Attack'), ('codec', 'Codec')]:
    merged = None
    for model_name, result in RESULTS_ASV5.items():
        table = grouped_eer_from_result(result, group_col)
        table = table.rename(columns={'eer': model_name})
        merged = table if merged is None else merged.merge(table, on=[group_col, 'n_spoof'], how='outer')

    if merged is None or merged.empty:
        continue

    ASV5_GROUP_TABLES[group_col] = merged
    display(merged)
    merged.to_csv(f'{ASV5_RESULTS_DIR}/eer_by_{group_col}_{ASV5_CONFIG["split"]}.csv', index=False)

    model_cols = [c for c in merged.columns if c not in (group_col, 'n_spoof')]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(merged))
    width = 0.8 / max(1, len(model_cols))
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    for i, col in enumerate(model_cols):
        offset = (i - len(model_cols) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], merged[col], width, label=col, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(merged[group_col], rotation=45, ha='right')
    ax.set_ylabel('EER (%)')
    ax.set_title(f'ASVspoof 5 {ASV5_CONFIG["split"]}: EER by {title}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{ASV5_RESULTS_DIR}/eer_by_{group_col}_{ASV5_CONFIG["split"]}.png', dpi=150, bbox_inches='tight')
    plt.show()
"""


PART7_CELL_74 = r"""# ═══════════════════════════════════════════
# 7.4  Cross-dataset comparison: ASVspoof 2019 vs 2021 vs 5
# ═══════════════════════════════════════════
rows = []
all_models = sorted(set(ALL_RESULTS) | set(RESULTS_2021) | set(RESULTS_ASV5))

for model_name in all_models:
    e19 = ALL_RESULTS.get(model_name, {}).get('eer', float('nan'))
    e21 = RESULTS_2021.get(model_name, {}).get('eer', float('nan'))
    e5 = RESULTS_ASV5.get(model_name, {}).get('eer', float('nan'))
    rows.append({
        'model': model_name,
        'asv19_la_eer': e19,
        'asv21_df_eer': e21,
        f'asv5_{ASV5_CONFIG["split"]}_eer': e5,
        'delta_21_minus_19': e21 - e19 if pd.notna(e19) and pd.notna(e21) else float('nan'),
        'delta_5_minus_19': e5 - e19 if pd.notna(e19) and pd.notna(e5) else float('nan'),
    })

cross_dataset_df = pd.DataFrame(rows).sort_values('model')
display(cross_dataset_df)
cross_dataset_df.to_csv(f'{ASV5_RESULTS_DIR}/cross_dataset_summary_{ASV5_CONFIG["split"]}.csv', index=False)

def hardest_group(table, model_name):
    if table is None or model_name not in table.columns or table.empty:
        return float('nan')
    return float(table[model_name].max())

attack_table_2019 = globals().get('eer_by_attack_2019')
codec_table_2021 = globals().get('eer_by_codec')
vocoder_table_2021 = globals().get('eer_by_vocoder')
attack_table_5 = ASV5_GROUP_TABLES.get('attack')
codec_table_5 = ASV5_GROUP_TABLES.get('codec')

profile_rows = []
for model_name in all_models:
    score_2019 = f'score_{model_name}'
    score_2021 = f'score_{model_name}'
    profile_rows.append({
        'model': model_name,
        'asv19_overall': ALL_RESULTS.get(model_name, {}).get('eer', float('nan')),
        'asv19_hardest_attack': hardest_group(attack_table_2019, score_2019) if attack_table_2019 is not None else float('nan'),
        'asv21_overall': RESULTS_2021.get(model_name, {}).get('eer', float('nan')),
        'asv21_hardest_codec': hardest_group(codec_table_2021, score_2021) if codec_table_2021 is not None else float('nan'),
        'asv21_hardest_vocoder': hardest_group(vocoder_table_2021, score_2021) if vocoder_table_2021 is not None else float('nan'),
        f'asv5_{ASV5_CONFIG["split"]}_overall': RESULTS_ASV5.get(model_name, {}).get('eer', float('nan')),
        f'asv5_{ASV5_CONFIG["split"]}_hardest_attack': hardest_group(attack_table_5, model_name),
        f'asv5_{ASV5_CONFIG["split"]}_hardest_codec': hardest_group(codec_table_5, model_name),
    })

difficulty_profile_df = pd.DataFrame(profile_rows).sort_values('model')
display(difficulty_profile_df)
difficulty_profile_df.to_csv(f'{ASV5_RESULTS_DIR}/difficulty_profile_{ASV5_CONFIG["split"]}.csv', index=False)

for name, data in RESULTS_ASV5.items():
    safe = name.replace('+', '_').replace(' ', '_')
    meta_df = pd.DataFrame(data['meta'])
    np.savez(
        f'{ASV5_RESULTS_DIR}/scores_asvspoof5_{ASV5_CONFIG["split"]}_{safe}.npz',
        scores=data['scores'],
        labels=data['labels'],
        utt_ids=meta_df.get('utt_id', pd.Series(dtype=str)).astype(str).values,
        attacks=meta_df.get('attack', pd.Series(dtype=str)).astype(str).values,
        codecs=meta_df.get('codec', pd.Series(dtype=str)).astype(str).values,
        eer=data['eer'],
    )

print(f'💾 ASVspoof 5 artifacts saved to {ASV5_RESULTS_DIR}/')
"""


def insert_part7(nb: dict) -> bool:
    joined_sources = ["".join(cell.get("source", [])) for cell in nb["cells"]]
    if any("## Part 7: ASVspoof 5 Evaluation & Comparison to ASVspoof 2019/2021" in src for src in joined_sources):
        return False

    insert_idx = next(
        i for i, src in enumerate(joined_sources) if "## Part 8: EchoFake Evaluation (2025)" in src
    )
    new_cells = [
        markdown_cell(PART7_MARKDOWN),
        code_cell(PART7_CELL_70),
        code_cell(PART7_CELL_71),
        code_cell(PART7_CELL_72),
        code_cell(PART7_CELL_73),
        code_cell(PART7_CELL_74),
    ]
    nb["cells"][insert_idx:insert_idx] = new_cells
    return True


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    exported = export_part6_outputs(nb)
    inserted = insert_part7(nb)
    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Exported {len(exported)} Part 6 artifacts to {EXPORT_DIR.as_posix()}")
    print("Inserted Part 7 cells" if inserted else "Part 7 cells already present")


if __name__ == "__main__":
    main()

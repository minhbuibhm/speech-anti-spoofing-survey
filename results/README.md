# Results

Evaluation results and model checkpoints from the anti-spoofing model comparison study.
All scores are produced by models trained/evaluated on Kaggle (GPU: P100/T4).

---

## Directory Structure

```
results/
├── asvspoof19/
│   └── results.pkl                           # eval scores for 4 models on ASVspoof 2019 LA
├── asvspoof21/
│   └── results.pkl                           # eval scores for 2 models on ASVspoof 2021 DF
├── asvspoof5/
│   └── results.pkl                           # eval scores for 4 models on ASVspoof 5 (2024)
├── in_the_wild/
│   └── results.pkl                           # eval scores for 6 models on In-the-Wild (pending)
├── checkpoints/
│   └── lfcc_lcnn/
│       ├── lfcc_lcnn.pth                     # trained weights (~407 KB)
│       └── lfcc_lcnn_train_summary.json      # training history
└── README.md
```

---

## Eval Results Format

Each `results.pkl` is a Python `dict` with the following structure:

```python
{
    "<model_name>": {
        "eer":    float,          # Equal Error Rate (%), lower is better
        "scores": np.ndarray,     # shape (N,), float64 — model output probabilities
        "labels": np.ndarray,     # shape (N,), int64  — 1=bonafide, 0=spoof
    },
    ...
}
```

Load with:
```python
import pickle
with open("results/asvspoof19/results.pkl", "rb") as f:
    results = pickle.load(f)
```

---

## `asvspoof19/results.pkl`

Evaluation on **ASVspoof 2019 LA** eval set.
- Total records: **71,237 utterances**
- Breakdown: 7,355 bonafide + 63,882 spoof
- 4 models evaluated

| Key | Model | EER (%) |
|-----|-------|---------|
| `AASIST` | End-to-end graph attention | 4.59 |
| `AASIST-L` | Lightweight AASIST | 6.74 |
| `AASIST3` | Wav2Vec2 + KAN + AASIST | 20.83 |
| `LFCC+LCNN` | Hand-crafted features + CNN | 19.64 |

---

## `asvspoof21/results.pkl`

Evaluation on **ASVspoof 2021 DF** eval set.
- Total records: **458,868 utterances**
- Breakdown: 16,977 bonafide + 441,891 spoof
- 2 models evaluated so far (AASIST-L and AASIST3 pending)

| Key | Model | EER (%) |
|-----|-------|---------|
| `AASIST` | End-to-end graph attention | 17.70 |
| `LFCC+LCNN` | Hand-crafted features + CNN | 33.81 |

> The large EER increase from 2019 → 2021 is expected: ASVspoof 2021 DF applies
> lossy codec compression which destroys the spectral artifacts that both models rely on.

---

## `asvspoof5/results.pkl`

Evaluation on **ASVspoof 5 (2024)** Track 1 eval set.
- Total records: **140,950 utterances**
- Breakdown: 31,334 bonafide + 109,616 spoof
- 4 models evaluated

| Key | Model | EER (%) |
|-----|-------|---------|
| `AASIST` | End-to-end graph attention | 37.81 |
| `AASIST-L` | Lightweight AASIST | 39.47 |
| `AASIST3` | Wav2Vec2 + KAN + AASIST | 19.03 |
| `LFCC+LCNN` | Hand-crafted features + CNN | 22.60 |

> ASVspoof 5 is the hardest generalization test: attacks are more diverse and
> include real-world codec/compression conditions. AASIST and AASIST-L degrade
> severely (EER near 40%) while AASIST3's SSL front-end provides more robustness.

---

## `in_the_wild/results.pkl`

Evaluation on **In-the-Wild** (Müller et al., 2022 — "Does Audio Deepfake Detection Generalize?").
- Total records: **31,779 utterances**
- Breakdown: 11,816 bonafide + 19,963 spoof
- 58 speakers (celebrities and politicians)
- 6 models to be evaluated

| Key | Model | EER (%) |
|-----|-------|---------|
| `AASIST` | End-to-end graph attention | pending |
| `AASIST-L` | Lightweight AASIST | pending |
| `AASIST3` | Wav2Vec2 + KAN + AASIST | pending |
| `LFCC+LCNN` | Hand-crafted features + CNN | pending |
| `XLS-R+AASIST` | XLS-R 300M + AASIST back-end | pending |
| `XLS-R+Nes2Net` | XLS-R 300M + Nes2Net-X back-end | pending |

> In-the-Wild is the primary cross-domain generalization probe: audio scraped from
> social media with unknown synthesis pipelines, real-world noise, and mixed codecs.
> None of the models were trained on this distribution.

---

## `checkpoints/lfcc_lcnn/`

Weights from training LFCC+LCNN from scratch on ASVspoof 2019 LA train set.

### `checkpoints/lfcc_lcnn.pth`

PyTorch state dict (~407 KB). Load with:
```python
import torch
model.load_state_dict(torch.load("results/model_weights/lfcc_lcnn/checkpoints/lfcc_lcnn.pth"))
```

### `checkpoints/lfcc_lcnn_train_summary.json`

Training history — 10 epochs, batch size 64:

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.1365 | 94.77% |
| 5 | 0.0162 | 99.46% |
| 10 | 0.0047 | 99.88% |

> High train accuracy does not reflect eval EER (19.64% on 2019 LA, 33.81% on 2021 DF)
> — indicates overfitting to training codec conditions.

---

## Dataset Notes

- **ASVspoof 2019 LA**: clean digital injection attacks (TTS/VC), standard benchmark
- **ASVspoof 2021 DF**: same attacks post-processed through lossy codecs — harder generalization test
- Protocol label: `bonafide` → 1, `spoof` → 0
- 2021 DF label is at field index 5 in the trial metadata (not the last field)

---

## Survey Evaluation Datasets

This survey uses single-utterance countermeasure (CM) evaluation where each audio file
receives one bonafide score. The common binary convention is:

- `bonafide` -> label `1`
- `spoof` -> label `0`
- model score -> bonafide probability

### ASVspoof 2019 LA

ASVspoof 2019 LA is the base benchmark for logical access spoofing attacks. The survey
evaluates the LA eval split using `ASVspoof2019.LA.cm.eval.trl.txt` and
`ASVspoof2019_LA_eval/flac`.

High-level structure:

- `ASVspoof2019_LA_train`
- `ASVspoof2019_LA_dev`
- `ASVspoof2019_LA_eval`
- `ASVspoof2019_LA_cm_protocols`

The CM protocol row format is `speaker utt_id - attack key`, where `key` is
`bonafide` or `spoof`.

### ASVspoof 2021 DF

ASVspoof 2021 DF is used as a harder cross-dataset generalization test. It reuses
attacks under lossy codec and compression conditions. The survey evaluates the DF eval
set only.

High-level structure:

- `ASVspoof2021_DF_eval_part00`
- `ASVspoof2021_DF_eval_part01`
- `ASVspoof2021_DF_eval_part02`
- `DF-keys-full/keys/DF/CM/trial_metadata.txt`

The CM label is field index `5` in `trial_metadata.txt`, not the last field.

### ASVspoof 5

ASVspoof 5 contains three splits and two tracks. The survey currently evaluates
**Track 1 only**, because Track 1 is the standalone CM task and matches the existing
`results.pkl` format.

Splits:

- `train`: training/tuning split, audio prefix `flac_T`
- `dev`: development split, audio prefix `flac_D`
- `eval`: official evaluation split, audio prefix `flac_E`

Track 1:

- Standalone CM task: `audio -> bonafide/spoof score`
- Protocol files:
  - `ASVspoof5.train.tsv`
  - `ASVspoof5.dev.track_1.tsv`
  - `ASVspoof5.eval.track_1.tsv`
- Track 1 row format:
  `SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`
- `KEY` is the binary CM label.

Track 2:

- Spoofing-aware speaker verification (SASV), not standalone CM.
- Protocol files:
  - `ASVspoof5.dev.track_2.enroll.tsv`
  - `ASVspoof5.dev.track_2.trial.tsv`
  - `ASVspoof5.eval.track_2.enroll.tsv`
  - `ASVspoof5.eval.track_2.trial.tsv`
- Task shape: `enrollment utterance + trial utterance -> accept/reject score`.

Track 2 requires speaker-verification logic and a different output schema, so it is not
included in the current `results.pkl` survey format.

### In-the-Wild

In-the-Wild (Muller et al., 2022, "Does Audio Deepfake Detection Generalize?") is a
real-world deepfake corpus scraped from public social media. It covers 58 celebrity and
politician speakers with unknown synthesis pipelines, mixed codecs, and real-world noise
conditions. It is used as a cross-domain generalization probe: no model in this survey
was trained on it.

- **Total**: 31,779 utterances — 11,816 bonafide + 19,963 spoof
- **Kaggle slug**: `abdallamohamed312/in-the-wild-audio-deepfake`

High-level structure:

- `meta.csv` — metadata with columns `file`, `speaker`, `label`
- flat audio directory (`release_in_the_wild/` or `audio/`) of WAV files at 16 kHz

Protocol format: `meta.csv` row with `label='bona-fide'` maps to bonafide (1); any
other value (e.g. `'spoof'`) maps to spoof (0). The parser resolves audio paths by
relative path, filename, and stem to tolerate Kaggle layout variations.

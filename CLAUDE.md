# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an internship research project comparing speech anti-spoofing / deepfake detection models across datasets. The goal is to reproduce EER results, analyze strengths and weaknesses of each approach, then propose an improvement.

**Current status:** Running models on ASVspoof 2021 DF eval (codec-heavy) to compare against ASVspoof 2019 LA results.

All experiments run on **Kaggle** (GPU: P100/T4). The single working file is `survey.ipynb`, which is synced to the user's Kaggle account — all code and cell outputs reflect actual Kaggle runs.

## Files

- `survey.ipynb` — main Kaggle notebook with all experiments
- `docs.md` — survey notes, SOTA tables, paper summaries, experiment plan
- `todo.md` — task checklist (source of truth for progress)
- `all_results_asvspooft_19.pkl` — saved scores from ASVspoof 2019 LA runs (loaded in Part 4/5 to avoid re-inference)

## Notebook Structure (`survey.ipynb`)

| Part | Content |
|------|---------|
| 0 | Common setup: GPU check, Kaggle dataset paths, utility functions, copy to local SSD |
| 1 | AASIST & AASIST-L — clone repo, load pretrained weights, evaluate on ASVspoof 2019 LA |
| 2 | AASIST3 (Wav2Vec2 + KAN) — load from HuggingFace `MTUCI/AASIST3`, evaluate |
| 3 | LFCC + LCNN — define and train from scratch (~10–15 min), evaluate |
| 4 | Results comparison — EER table, score distribution plots, save all scores to `.pkl` |
| 5 | ASVspoof 2021 DF evaluation — reload models, run on 2021 DF, compare to 2019 results |

## Models Being Compared

| Model | Type | Params | Expected EER (ASV2019 LA) |
|-------|------|--------|--------------------------|
| LFCC + LCNN | Hand-crafted + CNN | ~60K | ~5–10% |
| AASIST | End-to-end GAT | ~297K | ~0.83% |
| AASIST-L | End-to-end GAT (light) | ~85K | ~0.9–1% |
| AASIST3 | Wav2Vec2 + KAN + AASIST | ~300M | ~0.5–1% |

## Kaggle Dataset Paths

```python
# ASVspoof 2019 LA
ASV19_BASE = '/kaggle/input/...'   # see Cell 3 for actual path

# ASVspoof 2021 DF eval (3 parts symlinked into one dir)
DF_FLAC_DIR = '/kaggle/working/asvspoof2021_DF_flac'  # ~458k files
DF_TRIAL    = '/kaggle/input/.../DF-keys-full/keys/DF/CM/trial_metadata.txt'
LA21_TRIAL  = '/kaggle/input/.../LA-keys-full/keys/LA/CM/trial_metadata.txt'
```

## Protocol File Formats

**ASVspoof 2019 LA** — space-separated, label at index 4:
```
LA_0079 LA_E_1000147 - A09 spoof
LA_0079 LA_E_1000215 - - bonafide
```

**ASVspoof 2021 DF** — space-separated, label at **index 5** (NOT the last field):
```
LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -
TEF2    DF_E_2000013 low_m4a vcc2020   Task1-team20 spoof notrim eval ...
```
Key gotcha: `parts[-1]` is `-`, not the label. Always use `parts[5]`.

## Evaluation

Primary metric is **EER (Equal Error Rate)** — lower is better. Computed with `sklearn` or `scipy` from model scores vs ground-truth labels. Results are compared across:
1. ASVspoof 2019 LA (clean, standard benchmark)
2. ASVspoof 2021 DF (codec-compressed, harder generalization test)

The expected finding is large EER degradation on 2021 DF for all models, especially LFCC+LCNN, motivating the need for codec-robust approaches.

## Key Domain Concepts

- **Bonafide** = genuine speech (label 1), **Spoof** = synthetic/manipulated (label 0)
- **LA (Logical Access)**: digital injection attacks (TTS/VC)
- **DF (Deepfake)**: LA attacks processed through lossy codecs — destroys spectral artifacts that hand-crafted features rely on
- **SSL front-ends** (WavLM, Wav2Vec2, XLS-R): pre-trained large models used as feature extractors; WavLM Large is current best due to denoising pre-training objective
- **AASIST back-end**: heterogeneous graph attention over spectral + temporal domains; standard back-end when paired with SSL front-ends

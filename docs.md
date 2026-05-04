# Internship 1 Notes: Speech Deepfake Detection

## TODO List

- [x] Read the main survey papers on speech deepfake detection / audio anti-spoofing to build a field overview.
- [x] Read core architecture papers: RawNet2, AASIST, SSL+AASIST, WavLM, XLS-R, AASIST3, Nes2Net.
- [x] Survey key datasets/challenges: ASVspoof 2019 LA, ASVspoof 2021 DF, ASVspoof 5, In-the-Wild, MLAAD, CodecFake, EchoFake, ADD, PartialSpoof.
- [x] Set up evaluation environment and organize pretrained checkpoints/datasets on Kaggle.
- [x] Run/reproduce models on ASVspoof 2019 LA and record EER.
  - [x] LFCC + LCNN
  - [x] AASIST
  - [x] AASIST-L
  - [x] AASIST3
  - [x] XLS-R + AASIST
  - [x] XLS-R + Nes2Net
- [x] Run/reproduce available models on ASVspoof 2021 DF, ASVspoof 5 Track 1, and In-the-Wild.
- [x] Compare results across models and datasets.
- [x] Synthesize high-level strengths/weaknesses: hand-crafted features vs raw waveform models vs SSL front-ends.
- [x] Draft `report/report.md`.
- [x] Convert report content into LaTeX chapters.
- [x] Update `latex/main.tex`, `Chapter1.tex` to `Chapter4.tex`, `refs.bib`, and `abbreviation-list.tex`.
- [x] Replace `\ac{...}` usages with plain abbreviations in LaTeX content.
- [x] Fix Chapter 1 pipeline figure to avoid TikZ `right=of` build errors.
- [ ] Build final PDF on a machine with LaTeX installed and resolve remaining warnings.
- [ ] Finish ASVspoof 5 full eval split, not only dev/metadata-rich split.
- [ ] Complete detailed ASVspoof 5 error analysis:
  - [ ] Score distributions
  - [ ] EER by `attack_tag`
  - [ ] EER by `codec` / `codec_q`
  - [ ] Confident errors
  - [ ] Failure overlap between models
- [ ] Add a Nes2Net/no-XLS-R or equivalent baseline to isolate SSL contribution.
- [ ] Consider additional robustness experiments: codec augmentation, WavLM/XLS-R comparison, score fusion.
- [ ] Consider new datasets after protocol verification: CodecFake/CodecFake+, MLAAD, EchoFake, PartialSpoof.
- [ ] Prepare final Internship 1 presentation slides.

---

## Abstract

Speech Deepfake Detection (SDD) is the task of distinguishing genuine speech (*bonafide*) from synthesized or manipulated speech (*spoof*). This note summarizes the high-level content of the Internship 1 report: a survey of the SDD problem, datasets, model families, evaluation metrics, reproduced experiments, and follow-up TODOs.

The current report reproduces six models (LFCC+LCNN, AASIST, AASIST-L, AASIST3, XLS-R+AASIST, XLS-R+Nes2Net) on four datasets (ASVspoof 2019 LA, ASVspoof 2021 DF, ASVspoof 5 Track 1, In-the-Wild). The main observation is that SSL-based front-ends, especially XLS-R+Nes2Net, show smaller generalization gaps than hand-crafted or small raw-waveform systems under codec-heavy, modern-attack, and in-the-wild conditions.

---

## 1. Survey -- Speech Anti-Spoofing and Deepfake Detection

### 1.1 Motivation

Modern TTS, VC, diffusion, and neural codec systems can produce highly realistic cloned voices. This enables useful applications but also creates risks such as voice fraud, impersonation, and misinformation. Robust automatic detection is needed because clean benchmark performance does not guarantee real-world reliability.

### 1.2 Attack Taxonomy

- **Logical Access (LA):** digital injection of TTS/VC audio into a system.
- **Physical Access (PA) / replay:** spoofed or recorded speech replayed through speakers/microphones.
- **Deepfake / in-the-wild:** audio from social media or public sources, often with codec, noise, trimming, and missing metadata.
- **Adversarial / neural codec attacks:** attacks using perturbations or modern codec-based generation, often producing artifacts different from older vocoder-based datasets.
- **Partial spoof:** only a segment of an utterance is manipulated; requires localization or segment-level evaluation.

### 1.3 Spoofing in the Context of Speaker Verification

Automatic Speaker Verification (ASV) checks whether the speaker matches a claimed identity. A countermeasure (CM) checks whether the input speech is genuine or spoofed. The Internship 1 report focuses on standalone CM scoring, so EER is used as the main metric.

### 1.4 Datasets and Challenges

Main reproduced datasets:

| Dataset | Role in Report | Key Stress Test |
| --- | --- | --- |
| ASVspoof 2019 LA | Clean baseline benchmark | TTS/VC in controlled condition |
| ASVspoof 2021 DF | Codec-heavy benchmark | Lossy codec robustness |
| ASVspoof 5 Track 1 | Modern ASVspoof benchmark | Modern attacks, metadata-rich dev split |
| In-the-Wild | Real-world probe | Domain shift and label/noise uncertainty |

Referenced/future datasets:

- **MLAAD:** multilingual TTS evaluation.
- **CodecFake / CodecFake+:** neural codec / codec-based speech generation.
- **EchoFake:** replay-aware practical SDD.
- **ADD 2022/2023:** broader challenge tracks including low-quality and partial fake.
- **PartialSpoof:** utterance-level and segment-level partially spoofed audio.

### 1.5 Detection Methods

General pipeline:

`audio -> front-end representation -> back-end model -> score -> decision`

Main model families:

- **Hand-crafted features + classifier:** LFCC/CQCC/MFCC + GMM/SVM/LCNN. Lightweight but sensitive to dataset-specific artifacts.
- **Raw waveform / end-to-end:** RawNet2, AASIST, AASIST-L. Compact and strong on ASVspoof 2019 LA, but can degrade under unseen attacks/codecs.
- **SSL front-end + anti-spoofing back-end:** Wav2Vec2, XLS-R, WavLM, HuBERT with AASIST/Nes2Net/MHFA-style back-ends. Stronger generalization, higher compute.
- **System-level robustness:** augmentation, codec-aware training, calibration, ensemble/fusion.

### 1.6 Evaluation Metrics

- **EER:** main metric in the report; threshold where FAR and FRR are approximately equal.
- **min t-DCF:** useful when evaluating CM inside an ASV pipeline with cost model and ASV scores.
- **ROC/DET curves:** useful for threshold trade-off analysis.
- **Calibration metrics:** future work if fixed-threshold deployment is considered.

### 1.7 Open Challenges and Future Directions

- Cross-domain generalization from clean benchmark to real-world audio.
- Robustness to codec, compression, replay, and post-processing.
- Modern neural codec / diffusion / adversarial attacks.
- Multilingual and demographic coverage.
- Practical deployment constraints: latency, VRAM, edge devices, calibration.
- Error analysis that explains *why* each model fails, not only total EER.

---

## 2. State-of-the-Art Methods

### 2.1 AASIST

AASIST uses graph attention over spectral and temporal branches from raw waveform input. It is compact and strong on ASVspoof 2019 LA, but the reproduced results show weaker generalization on ASVspoof 5 and In-the-Wild.

### 2.2 RawNet2

RawNet2 is an important raw waveform baseline using SincConv/residual blocks. It is mostly used in the report as historical context for end-to-end anti-spoofing.

### 2.3 SSL Front-end + Back-end Paradigm

This is the strongest direction in the current experiments. XLS-R+AASIST and XLS-R+Nes2Net use large pretrained speech representations before anti-spoofing back-ends. XLS-R+Nes2Net has the best reproduced stability across datasets.

### 2.4 ASVspoof 5 Systems

ASVspoof 5 motivates the next phase of analysis because it includes crowdsourced speech, modern attacks, and adversarial/neural encoding conditions. The current report uses Track 1 dev split for preliminary analysis; full eval remains TODO.

### 2.5 Emerging Directions

- Codec-aware detection/training.
- WavLM/XLS-R/Whisper-like foundation front-ends.
- Score-level or feature-level fusion.
- Partial spoof localization.
- Source attribution and multi-task detection.

---

## 3. Detailed Paper Summaries, Result Tables, and Data Sources

### 3.1 Core Paper Groups

- **Surveys:** establish dataset/model/metric landscape and open challenges.
- **ASVspoof papers:** define challenge protocols and benchmark evolution.
- **AASIST/RawNet2:** represent compact end-to-end waveform detectors.
- **SSL anti-spoofing papers:** motivate XLS-R/WavLM front-end usage.
- **Nes2Net/AASIST3:** directly related to reproduced models.
- **CodecFake/MLAAD/EchoFake/PartialSpoof:** guide future dataset expansion.

### 3.2 Reproduced Result Table

EER (%), lower is better:

| Model | ASV19 LA | ASV21 DF | ASV5 | In-the-Wild |
| --- | ---: | ---: | ---: | ---: |
| LFCC + LCNN | 19.64 | 33.81 | 22.60 | 70.23 |
| AASIST | 4.59 | 17.70 | 37.81 | 41.79 |
| AASIST-L | 6.74 | 19.13 | 39.47 | 45.28 |
| AASIST3 | 20.83 | 29.18 | 19.03 | 40.12 |
| XLS-R + AASIST | 1.17 | - | 2.55 | - |
| XLS-R + Nes2Net | 0.45 | 2.93 | 1.80 | 5.57 |

Generalization gap vs ASVspoof 2019 LA:

| Model | ASV21 DF - ASV19 | ASV5 - ASV19 | ITW - ASV19 |
| --- | ---: | ---: | ---: |
| LFCC + LCNN | +14.17 | +2.96 | +50.59 |
| AASIST | +13.11 | +33.22 | +37.20 |
| AASIST-L | +12.39 | +32.73 | +38.54 |
| AASIST3 | +8.35 | -1.80 | +19.29 |
| XLS-R + AASIST | - | +1.38 | - |
| XLS-R + Nes2Net | +2.48 | +1.35 | +5.12 |

### 3.3 Data Sources

Primary data sources used or referenced:

- ASVspoof official datasets/challenge pages.
- Kaggle datasets for In-the-Wild and experiment packaging.
- Public model checkpoints: AASIST/AASIST-L, AASIST3, XLS-R, XLS-R+AASIST, XLS-R+Nes2Net.
- Future candidates: CodecFake/CodecFake+, MLAAD, EchoFake, PartialSpoof.

---

## 4. Experiment Plan -- Running Pretrained Models

### 4.1 Objective

Use pretrained or publicly reproducible models to compare practical EER across clean, codec-heavy, modern-attack, and in-the-wild datasets.

### 4.2 Environment

Current experiments were designed around Kaggle GPU sessions (P100/T4). SSL models require substantially more VRAM and runtime than LFCC+LCNN/AASIST-style models.

### 4.3 Dataset Preparation

Each dataset needs:

- Audio path resolver.
- Protocol/metadata parser.
- Ground-truth label mapping: `bonafide = 1`, `spoof = 0`.
- Consistent score output format.

### 4.4 Models to Run

Completed in Internship 1:

- LFCC + LCNN
- AASIST
- AASIST-L
- AASIST3
- XLS-R + AASIST
- XLS-R + Nes2Net

Still useful to add:

- Nes2Net without XLS-R or equivalent non-SSL baseline.
- WavLM-based detector under same protocol.
- Codec-aware or fusion variant after error analysis.

### 4.5 Evaluation Protocol

For every model/dataset pair:

1. Produce utterance-level bonafide scores.
2. Store `scores`, `labels`, and `eer` in `results/<dataset>/results.pkl`.
3. Compute EER consistently from score/label arrays.
4. Compare total EER and generalization gap.
5. For ASVspoof 5, group errors by metadata when available.

### 4.6 Expected Outcome

The final report should not only rank models by EER, but also explain which conditions make each family fail:

- LFCC+LCNN: likely sensitive to codec/domain shift.
- AASIST/AASIST-L: strong clean benchmark baseline, weaker modern attack robustness.
- AASIST3: behavior differs across ASVspoof versions, likely due to training distribution.
- XLS-R-based systems: strongest current generalization, but expensive and needs isolation study.

---

## References

See `latex/refs.bib` for the current report bibliography. Main reference groups:

- ASVspoof 2019, ASVspoof 2021, ASVspoof 5.
- In-the-Wild audio deepfake detection.
- RawNet2, AASIST, SSL+AASIST, AASIST3, Nes2Net.
- Wav2Vec2, XLS-R, WavLM.
- CodecFake, CodecFake+, MLAAD, EchoFake, ADD, PartialSpoof.

---

## Files

- Draft report: `report/report.md`
- LaTeX entry: `latex/main.tex`
- Chapters: `latex/chapters/Chapter1.tex` to `latex/chapters/Chapter4.tex`
- References: `latex/refs.bib`
- Acronyms: `latex/chapters/abbreviation-list.tex`

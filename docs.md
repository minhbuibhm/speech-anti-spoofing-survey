

[**TODO List	2**](#todo-list)

[**Abstract	2**](#abstract)

[**1\. Survey \-- Speech Anti-Spoofing and Deepfake Detection	3**](#1.-survey----speech-anti-spoofing-and-deepfake-detection)

[1.1 Motivation	3](#1.1-motivation)

[1.2 Attack Taxonomy	3](#1.2-attack-taxonomy)

[1.3 Spoofing in the Context of Speaker Verification	3](#1.3-spoofing-in-the-context-of-speaker-verification)

[1.4 Datasets and Challenges	3](#1.4-datasets-and-challenges)

[1.5 Detection Methods	4](#1.5-detection-methods)

[1.6 Evaluation Metrics	5](#1.6-evaluation-metrics)

[1.7 Open Challenges and Future Directions	5](#1.7-open-challenges-and-future-directions)

[**2\. State-of-the-Art Methods \- To be updated	6**](#2.-state-of-the-art-methods---to-be-updated)

[2.1 AASIST (Jung et al., ICASSP 2022\)	6](#2.1-aasist-\(jung-et-al.,-icassp-2022\))

[2.2 RawNet2 (Tak et al., ICASSP 2021\)	6](#2.2-rawnet2-\(tak-et-al.,-icassp-2021\))

[2.3 SSL Front-end \+ AASIST Back-end Paradigm (2022--present)	6](#2.3-ssl-front-end-+-aasist-back-end-paradigm-\(2022--present\))

[2.4 Top Systems in ASVspoof 5 (2024)	7](#2.4-top-systems-in-asvspoof-5-\(2024\))

[2.5 Emerging SOTA Directions	7](#2.5-emerging-sota-directions)

[**3\. Detailed Paper Summaries, Result Tables, and Data Sources \- To be updated	8**](#3.-detailed-paper-summaries,-result-tables,-and-data-sources---to-be-updated)

[3.1 Core Paper Summaries	8](#3.1-core-paper-summaries)

[3.2 Result Comparison Tables	8](#3.2-result-comparison-tables)

[3.3 Audio Demo Sources	10](#3.3-audio-demo-sources)

[**4\. Experiment Plan \- Running Pretrained Models \- To be updated	11**](#4.-experiment-plan---running-pretrained-models---to-be-updated)

[4.1 Objective	11](#4.1-objective)

[4.2 Hardware and Software Requirements	11](#4.2-hardware-and-software-requirements)

[4.3 Dataset Preparation	11](#4.3-dataset-preparation)

[4.4 Models to Run	11](#4.4-models-to-run)

[4.5 Evaluation Protocol	13](#4.5-evaluation-protocol)

[4.6 Expected Outcome	13](#4.6-expected-outcome)

[**References \- To be updated	14**](#references---to-be-updated)

## **TODO List** {#todo-list}

- [x] ~~Read the main survey paper (Li et al., 2024\) and the PMC survey (Zhang et al., 2025\) to build a solid overview of the field~~  
- [x] ~~Read AASIST, WavLM, and Wav2Vec2+AASIST papers to understand the SOTA architectures~~  
- [x] ~~Set up environment and pre-trained models, download ASVspoof 2019 LA dataset.~~  
      - [x] ~~Run AASIST (pretrained) on ASVspoof 2019 LA eval and record EER.~~   
      - [x] ~~Run Nes2Net (WavLM, pretrained) on ASVspoof 2019 LA eval and record EER.~~   
      - [x] ~~Run AASIST3 (HuggingFace, pretrained) on ASVspoof 2019 LA eval and record EER.~~   
      - [x] ~~Run ASVspoof 2021 baseline (LFCC+LCNN, pretrained) on ASVspoof 2019 LA eval and record EER~~  
- [ ] \[**in-progress**\] Re-run the above models on ASVspoof 2021 DF eval (codec-heavy) and/or In-the-Wild dataset  
- [ ] \[**in-progress**\] Compare results across models and datasets; identify where each model succeeds and fails  
      - [ ] Synthesize insights on pros/cons of each approach (hand-crafted vs end-to-end vs SSL-based)  
- [ ] Propose a more robust approach based on observed weaknesses (e.g., codec augmentation, SSL fine-tuning strategy, ensemble)  
      - [ ] Implement and quickly evaluate the proposed improvement  
- [ ] Prepare final presentation slides

---

## **Abstract** {#abstract}

Speech deepfake detection, also known as speech anti-spoofing, is the task of distinguishing genuine (bonafide) speech from synthetically generated or manipulated speech. With rapid advances in text-to-speech (TTS) and voice conversion (VC) systems, including recent zero-shot and codec-based models such as VALL-E and SoundStorm, the threat of realistic voice deepfakes has become critical. This document consolidates a survey-level overview of the field, a focused review of state-of-the-art (SOTA) methods, supporting materials including paper summaries and result tables, and a practical plan for reproducing key experiments. The intended audience is graduate-level, and the material is structured to support two 15-to-20-minute presentations: one survey-oriented and one SOTA-oriented.

---

## 

## **1\. Survey \-- Speech Anti-Spoofing and Deepfake Detection** {#1.-survey----speech-anti-spoofing-and-deepfake-detection}

### **1.1 Motivation** {#1.1-motivation}

Deepfake voices are increasingly realistic and accessible. Real-world incidents include financial fraud via CEO voice cloning and political misinformation campaigns. Human listeners perform poorly at distinguishing fake from real speech, with studies reporting accuracy around 73%, sometimes worse than random guessing. This motivates the need for robust automatic detection systems.

### **1.2 Attack Taxonomy** {#1.2-attack-taxonomy}

Speech spoofing attacks are categorized as follows:

**Logical Access (LA):** attacks injected digitally, without acoustic propagation.

* Text-to-Speech (TTS): autoregressive (Tacotron, VITS), non-autoregressive (FastSpeech), and LLM/codec-based (VALL-E, SoundStorm).  
* Voice Conversion (VC): disentanglement-based, GAN-based, diffusion-based.

**Physical Access (PA):** replay attacks through loudspeakers and microphones.

**Deepfake (DF):** similar to LA but processed through lossy codecs, simulating real-world media distribution.

**Partial Spoof:** only a segment within an utterance is replaced or manipulated.

**Adversarial Attack:** small perturbations added to fool detection systems while remaining imperceptible to humans.

### **1.3 Spoofing in the Context of Speaker Verification** {#1.3-spoofing-in-the-context-of-speaker-verification}

Automatic Speaker Verification (ASV) systems are vulnerable to spoofing. A countermeasure (CM) is a system designed to detect spoofing. Two deployment paradigms exist: standalone CM (binary bonafide/spoof decision) and Spoofing-Aware Speaker Verification (SASV), which integrates CM with ASV into a unified system.

### **1.4 Datasets and Challenges** {#1.4-datasets-and-challenges}

**ASVspoof Challenge Series:**

| Challenge | Year | Key Features |
| ----- | ----- | ----- |
| ASVspoof 2015 | 2015 | First edition, TTS/VC attacks only |
| ASVspoof 2017 | 2017 | Replay attacks (PA) |
| ASVspoof 2019 | 2019 | LA and PA tracks, VCTK-based, t-DCF metric |
| ASVspoof 2021 | 2021 | New DF track, codec/channel variation added |
| ASVspoof 5 | 2024 | Crowdsourced data (\~2000 speakers), adversarial attacks, a-DCF metric |

**Other Important Datasets:**

* In-the-Wild (ITW): celebrity/politician deepfakes collected from social media, with real-world noise.  
* MLAAD: 54 TTS models across 23 languages for multilingual evaluation.  
* CodecFake: focused on LLM-based/codec-based synthesis artifacts.  
* PartialSpoof: segment-level labels for localized deepfake detection.  
* FakeOrReal, WaveFake, TIMIT-TTS: supplementary benchmarks.

Common limitation: most datasets are English-dominant, recorded in clean conditions, with limited linguistic and acoustic diversity.

### **1.5 Detection Methods** {#1.5-detection-methods}

**General Pipeline:**

Raw audio \-\> Front-end (feature extraction) \-\> Back-end (classifier) \-\> Score \-\> Decision

**Generation 1: Hand-crafted Features (pre-2020)**

Spectral features such as MFCC, LFCC, CQT, and CQCC, combined with classifiers like GMM, SVM, or i-vector+PLDA. These are interpretable and lightweight but generalize poorly to unseen attacks.

**Generation 2: Deep Learning End-to-End (2019--2022)**

* RawNet2 (Tak et al., 2021): raw waveform input via SincConv, GRU encoder, fully connected output.  
* AASIST (Jung et al., 2022): heterogeneous graph attention over spectral and temporal domains. Only 297K parameters but achieved EER of 0.83% on ASVspoof 2019 LA.  
* Various ResNet and LCNN variants with data augmentation and margin-based losses.

**Generation 3: Self-Supervised Learning (SSL) Front-ends (2022--present)**

The dominant current paradigm. A pre-trained SSL model serves as the feature extractor, fine-tuned for anti-spoofing. Key SSL models include wav2vec 2.0, HuBERT, WavLM, XLS-R, and Whisper. WavLM consistently achieves the best results due to its denoising pre-training objective and gated relative position bias. Back-end options range from simple linear classifiers to MHFA pooling and AASIST-style graph networks. Results: EER below 1% on ASVspoof 2019 LA. Limitation: large models (300M+ parameters), expensive fine-tuning, and still struggling with unseen codecs and in-the-wild data.

**Optimization Techniques:**

Data augmentation (noise, codec simulation, RIR), loss functions (OC-Softmax, large-margin, contrastive), domain adaptation, ensemble/fusion, and continual/few-shot learning.

### **1.6 Evaluation Metrics** {#1.6-evaluation-metrics}

| Metric | Description |
| ----- | ----- |
| EER | Equal Error Rate: point where FAR equals FRR |
| min t-DCF | Tandem Detection Cost Function: evaluates CM in the context of ASV |
| a-DCF | Architecture-agnostic DCF: introduced in ASVspoof 5, single-threshold |
| t-EER | Tandem EER: parameter-free SASV evaluation |
| Range-based EER | For partial spoof: measures boundary-level accuracy |

### **1.7 Open Challenges and Future Directions** {#1.7-open-challenges-and-future-directions}

Key challenges: generalizability (lab-to-wild gap remains large), codec robustness, cross-language transfer, partial spoof detection, adversarial robustness, explainability, and real-time/edge deployment.

Emerging directions: foundation models and audio LLMs for detection, source attribution (identifying which model generated the deepfake), audio-visual multimodal detection, fairness and bias across demographics, privacy-preserving detection, and noise-aware detection.

---

## 

## **2\. State-of-the-Art Methods \- To be updated** {#2.-state-of-the-art-methods---to-be-updated}

### **2.1 AASIST (Jung et al., ICASSP 2022\)** {#2.1-aasist-(jung-et-al.,-icassp-2022)}

Heterogeneous graph attention network operating on spectral and temporal domains simultaneously. Uses a RawNet-style encoder (SincConv \+ ResBlocks) as front-end. Introduces the heterogeneous stacking graph attention layer (HS-GAL) and a max graph operation with competitive mechanism. Only 297K parameters (AASIST-L: 85K). Achieved EER of 0.83% and min t-DCF of 0.028 on ASVspoof 2019 LA eval. Became the standard baseline and a popular back-end when paired with SSL front-ends. Limitation: generalization degrades under unseen codec and noise conditions.

### **2.2 RawNet2 (Tak et al., ICASSP 2021\)** {#2.2-rawnet2-(tak-et-al.,-icassp-2021)}

Pioneered raw waveform end-to-end detection. SincConv layer learns filter banks from data, followed by ResBlocks and GRU. Competitive on ASVspoof 2019 but superseded by AASIST and SSL-based methods. Important as a historical milestone and still used as a component in larger systems.

### **2.3 SSL Front-end \+ AASIST Back-end Paradigm (2022--present)** {#2.3-ssl-front-end-+-aasist-back-end-paradigm-(2022--present)}

The dominant current approach. The SincConv front-end of AASIST is replaced with a pre-trained SSL model. Tak et al. (Odyssey 2022\) demonstrated XLS-R (300M) \+ AASIST, achieving major improvements on ASVspoof 2021\.

**Comparison of SSL front-ends (approximate, same back-end):**

| SSL Model | Params | ASVspoof 2019 LA EER | Notes |
| ----- | ----- | ----- | ----- |
| wav2vec 2.0 Base | 95M | \~1.5% | Baseline SSL |
| wav2vec 2.0 Large | 317M | \~0.8% | Better with scale |
| HuBERT Large | 317M | \~0.9% | Comparable to wav2vec2 |
| WavLM Large | 317M | \~0.3--0.5% | Best overall |
| XLS-R 300M | 300M | \~0.7% | Multilingual |

WavLM outperforms others because of its joint masked prediction and denoising pre-training, which captures both speaker information and noise robustness.

**Layer aggregation strategies:**

SSL models have multiple Transformer layers. Low/mid layers capture speaker information; top layers capture content. Learnable weighted sum and MHFA (Multi-Head Factorized Attentive pooling) are used to aggregate across layers. Key insight: if the SSL model is fine-tuned, a simple back-end suffices; if frozen, a deeper back-end is needed.

### **2.4 Top Systems in ASVspoof 5 (2024)** {#2.4-top-systems-in-asvspoof-5-(2024)}

ASVspoof 5 introduced crowdsourced data from \~2000 speakers, 32 attack algorithms including adversarial attacks, and new metrics (a-DCF, t-EER).

**USTC-KXDIGIT:** top performer on both tracks. Ensemble of multiple SSL models (wav2vec2, WavLM, HuBERT) with AASIST-like back-end, strong data augmentation, and score-level fusion.

**BUT System:** ResNet18 for closed condition, SSL+MHFA for open condition. Proposed effective priors for SASV logistic regression fusion. Provided analysis showing complex interaction between speaker and spoofing information.

**WavLM-based systems:** multiple teams used WavLM Large as the dominant front-end, with variations in back-end (weighted average, MHFA, ResNet-SA) and augmentation strategies. WavLM was the most popular front-end in the open condition.

### **2.5 Emerging SOTA Directions** {#2.5-emerging-sota-directions}

**Codec-aware detection:** LLM-based TTS (VALL-E, SoundStorm) uses neural codecs, producing different artifacts from vocoder-based systems. Models trained only on ASVspoof 2019 fail on CodecFake data. Codec augmentation during training is necessary.

**Partial deepfake detection and source attribution:** detecting and localizing fake segments within utterances (PartialSpoof dataset, range-based EER). Source attribution goes beyond binary classification to identify which generation model was used.

**Audio LLMs and foundation models:** DFALLM (2025) optimizes audio LLM components for generalizable multi-task detection. Noise-aware detection with SNR benchmarks is also emerging. These approaches are promising but computationally expensive.

---

## 

## **3\. Detailed Paper Summaries, Result Tables, and Data Sources \- To be updated** {#3.-detailed-paper-summaries,-result-tables,-and-data-sources---to-be-updated}

### **3.1 Core Paper Summaries** {#3.1-core-paper-summaries}

**Li et al., "A Survey on Speech Deepfake Detection," ACM Computing Surveys, 2024\.** Systematically analyzes over 200 papers. Covers the full detection pipeline: model architectures, optimization, generalizability, metrics, datasets, and open-source availability. Key finding: SSL-based front-ends (especially WavLM) are the current best; fine-tuning SSL is essential; the back-end complexity can be reduced if the front-end is fine-tuned.

**Zhang et al., "Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead," Sensors, 2025\.** Extends prior surveys with coverage of privacy-preserving detection, fairness, and explainability. Reports that human detection accuracy is approximately 73%. Highlights that codec compression destroys spoofing artifacts and that fairness across gender, age, and accent remains underexplored.

**Jung et al., "AASIST," ICASSP 2022\.** Proposes heterogeneous stacking graph attention over spectral and temporal domains. 297K parameters. EER 0.83% on ASVspoof 2019 LA. Lightweight variant AASIST-L at 85K parameters still outperforms most competing systems.

**Tak et al., "Wav2Vec2 \+ AASIST," Odyssey 2022\.** Replaces AASIST front-end with XLS-R (300M). Combined with data augmentation (RawBoost), achieves the lowest EER reported at the time on ASVspoof 2021 LA and DF. Demonstrates that SSL representations, even pre-trained on bonafide-only data, transfer well to spoofing detection when fine-tuned.

**Chen et al., "WavLM," IEEE JSTSP 2022\.** Self-supervised model with masked prediction and denoising objectives. Gated relative position bias. Pre-trained on LibriLight 60kh. Achieves SOTA on the SUPERB benchmark across 15 speech tasks. For anti-spoofing, WavLM Large consistently outperforms wav2vec2 and HuBERT.

**Wang et al., "ASVspoof 5," 2024\.** Fifth edition of the challenge. Crowdsourced data from \~2000 speakers in diverse conditions. 32 attack algorithms including adversarial attacks (first time). New metrics: a-DCF and t-EER. Baseline systems are significantly compromised; top submissions bring substantial improvements through SSL front-ends and ensemble strategies.

### **3.2 Result Comparison Tables** {#3.2-result-comparison-tables}

**Table 1: Evolution on ASVspoof 2019 LA (Eval Set)**

| Method | Front-end | Back-end | EER (%) | min t-DCF | Params |
| ----- | ----- | ----- | ----- | ----- | ----- |
| CQCC \+ GMM | CQCC | GMM | \~9.57 | \~0.237 | \~1M |
| LFCC \+ GMM | LFCC | GMM | \~8.09 | \~0.212 | \~1M |
| LFCC \+ LCNN | LFCC | LCNN | \~2.71 | \~0.059 | \~1M |
| RawNet2 | Raw (SincConv) | GRU+FC | \~5.13 | \~0.114 | \~25M |
| RawGAT-ST | Raw (SincConv) | GAT | \~1.06 | \~0.036 | \~0.5M |
| AASIST | Raw (SincConv) | HS-GAL | \~0.83 | \~0.028 | \~0.3M |
| XLS-R \+ AASIST | XLS-R 300M | AASIST | \~0.50 | \-- | \~300M |
| WavLM \+ MHFA | WavLM Large | MHFA | \~0.3--0.5 | \-- | \~320M |

**Table 2: ASVspoof 2021 DF (Eval Set) \-- Harder due to codec compression**

| Method | EER (%) | Notes |
| ----- | ----- | ----- |
| Baseline LFCC+LCNN | \~15--20% | Official baseline |
| RawNet2 | \~8--10% | Post-challenge |
| XLS-R \+ AASIST | \~2.85% | With data augmentation |
| WavLM ensemble | \~2--4% | ASVspoof 5-era systems |

**Table 3: Cross-Dataset Generalization (Trained on ASVspoof 2019\)**

| Model | ASVspoof19 LA EER | In-the-Wild EER |
| ----- | ----- | ----- |
| AASIST | \~0.83% | \~15--25% |
| Wav2Vec2 \+ AASIST | \~0.5% | \~10--15% |
| WavLM \+ back-end | \~0.3% | \~8--12% |
| Ensemble \+ augmentation | \~0.2% | \~5--8% |

Note: all EER values are approximate, compiled from multiple papers with varying experimental setups. Exact numbers should be verified against the original publications.

**Table 4: ASVspoof Challenge Timeline**

| Year | Challenge | Source Data | Attacks | Tracks | Primary Metric | Speakers |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 2015 | ASVspoof 2015 | VCTK | 10 TTS/VC | LA | EER | \~45 |
| 2017 | ASVspoof 2017 | RedDots | Replay | PA | EER | \~42 |
| 2019 | ASVspoof 2019 | VCTK | 19 TTS/VC \+ Replay | LA \+ PA | min t-DCF, EER | \~107 |
| 2021 | ASVspoof 2021 | VCTK \+ others | Same \+ codec/channel | LA \+ PA \+ DF | min t-DCF, EER | \~107 |
| 2024 | ASVspoof 5 | MLS (crowdsourced) | 32 \+ adversarial | CM \+ SASV | a-DCF, t-EER | \~2000 |

### **3.3 Audio Demo Sources** {#3.3-audio-demo-sources}

**ASVspoof Official Audio Examples:** [https://www.asvspoof.org/audio\_examples](https://www.asvspoof.org/audio_examples) \-- includes samples from ASVspoof 2019 LA with both human perception scores and AASIST machine scores. Recommended as the primary demo source.

**ASVspoof 2019 LA Dataset:** [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336) \-- free registration required. FLAC, 16kHz.

**ASVspoof 2021 DF Evaluation Set:** [https://zenodo.org/records/4835108](https://zenodo.org/records/4835108) \-- includes codec-compressed bonafide and spoofed speech.

**In-the-Wild Dataset:** available on Kaggle (search "In the Wild audio deepfake"). Celebrity/politician deepfakes from social media with real-world noise.

**Suggested demo scenarios for presentations:**

Scenario A ("Can you tell?"): play 3 audio clips, ask the audience to guess, reveal that human accuracy is around 73%.

Scenario B ("Lab vs Wild"): play a clean ASVspoof 2019 spoof sample, then an In-the-Wild sample, to illustrate the generalization gap.

Scenario C ("Codec effect"): play an uncompressed spoof sample, then the same sample after MP3/AAC compression, to show how codecs destroy detection artifacts.

---

## 

## **4\. Experiment Plan \- Running Pretrained Models \- To be updated** {#4.-experiment-plan---running-pretrained-models---to-be-updated}

### **4.1 Objective** {#4.1-objective}

Run four pretrained detection models on ASVspoof 2019 LA eval and optionally on ASVspoof 2021 DF or In-the-Wild, to obtain first-hand EER results and form practical observations about each approach's strengths and weaknesses.

### **4.2 Hardware and Software Requirements** {#4.2-hardware-and-software-requirements}

GPU with at least 8GB VRAM (16GB recommended for SSL models). Python 3.8--3.10, PyTorch 1.10+, torchaudio, HuggingFace transformers. Linux recommended.

### **4.3 Dataset Preparation** {#4.3-dataset-preparation}

Download ASVspoof 2019 LA from Edinburgh DataShare (free, academic use). Optionally download ASVspoof 2021 DF eval from Zenodo and In-the-Wild from Kaggle. All audio should be 16kHz mono.

### **4.4 Models to Run** {#4.4-models-to-run}

**Model A: AASIST (Official)**

| Property | Value |
| ----- | ----- |
| Repository | [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist) |
| Pretrained weights | Included in repo (AASIST and AASIST-L) |
| Parameters | 297K (AASIST), 85K (AASIST-L) |
| GPU memory | \~2GB |
| Expected EER | \~0.83% on ASVspoof 2019 LA eval |
| Setup difficulty | Easy |
| Run command | python main.py \--eval \--config ./config/AASIST.conf |

Why run this: the standard baseline that every paper compares against. Lightweight, fast, easy to set up.

**Model B: Nes2Net with WavLM front-end**

| Property | Value |
| ----- | ----- |
| Repository | [https://github.com/Liu-Tianchi/Nes2Net\_ASVspoof\_ITW](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW) |
| Pretrained weights | Google Drive links in README (multiple variants) |
| Front-end | WavLM Large |
| Parameters | \~300M+ |
| GPU memory | \~16GB |
| Expected EER | Below 0.5% on ASVspoof 2019 LA eval |
| Setup difficulty | Medium (requires fairseq installation) |

Why run this: published in IEEE T-IFS, pretrained checkpoints for ASVspoof 2019/2021 and In-the-Wild, represents the SSL-based SOTA.

**Model C: AASIST3 (Wav2Vec2 \+ KAN-enhanced AASIST)**

| Property | Value |
| ----- | ----- |
| Repository | [https://github.com/mtuciru/AASIST3](https://github.com/mtuciru/AASIST3) |
| HuggingFace | MTUCI/AASIST3 |
| Pretrained weights | HuggingFace Hub (load with from\_pretrained) |
| Front-end | Wav2Vec2 encoder |
| GPU memory | \~6--8GB |
| Setup difficulty | Medium |

Why run this: convenient HuggingFace integration, represents the KAN-enhanced variant, good for quick inference on individual files.

**Model D: ASVspoof 2021 Baseline (LFCC \+ LCNN)**

| Property | Value |
| ----- | ----- |
| Repository | [https://github.com/asvspoof-challenge/2021](https://github.com/asvspoof-challenge/2021) |
| Pretrained weights | Included |
| Parameters | \~1M |
| GPU memory | \~2GB |
| Expected EER | \~5--10% on ASVspoof 2019 LA eval |
| Setup difficulty | Easy |

Why run this: represents the traditional deep learning approach (hand-crafted features \+ CNN). Essential for showing the performance gap compared to end-to-end and SSL-based methods.

### **4.5 Evaluation Protocol** {#4.5-evaluation-protocol}

For each model, compute scores on the ASVspoof 2019 LA eval set using the provided protocols, then calculate EER using sklearn or the official scoring tools. Optionally repeat on ASVspoof 2021 DF eval and In-the-Wild to measure cross-dataset degradation.

### **4.6 Expected Outcome** {#4.6-expected-outcome}

A comparison table with self-reproduced EER values across models and datasets. This table directly supports the presentation by providing first-hand evidence of the progression from traditional methods (\~5--10% EER) to end-to-end models (\~0.83%) to SSL-based models (below 0.5%) on clean data, and the significant degradation on codec-heavy or in-the-wild data.

---

## 

## **References \- To be updated** {#references---to-be-updated}

\[1\] M. Li, Y. Ahmadiadli, and X.-P. Zhang, "A Survey on Speech Deepfake Detection," ACM Computing Surveys, 2024\. arXiv:2404.13914.

\[2\] B. Zhang, H. Cui, V. Nguyen, and M. Whitty, "Audio Deepfake Detection: What Has Been Achieved and What Lies Ahead," Sensors, vol. 25, no. 7, 2025\.

\[3\] J. Yi et al., "Audio Deepfake Detection: A Survey," arXiv:2308.14970, 2023\.

\[4\] J.-W. Jung et al., "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," in Proc. ICASSP, 2022\.

\[5\] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, "End-to-End Anti-Spoofing with RawNet2," in Proc. ICASSP, 2021\.

\[6\] H. Tak, M. Todisco, X. Wang, J.-W. Jung, J. Yamagishi, and N. Evans, "Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2Vec 2.0 and Data Augmentation," in Proc. Odyssey, 2022\. arXiv:2202.12233.

\[7\] S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE Journal of Selected Topics in Signal Processing, 2022\. arXiv:2110.13900.

\[8\] X. Wang et al., "ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale," in Proc. ASVspoof Workshop, 2024\. arXiv:2408.08739.

\[9\] J. Rohdin et al., "BUT Systems and Analyses for the ASVspoof 5 Challenge," in Proc. ASVspoof Workshop, 2024\.

\[10\] Y. Chen et al., "USTC-KXDIGIT System Description for ASVspoof5 Challenge," in Proc. ASVspoof Workshop, 2024\.

\[11\] T. Stourbe, V. Miara, T. Lepage, and R. Dehak, "Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection," in Proc. ASVspoof Workshop, 2024\. arXiv:2409.05032.

\[12\] K. Borodin et al., "AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection Using SSL Features," in Proc. ASVspoof Workshop, 2024\.

\[13\] T.-C. Liu et al., "Nes2Net: Nested Architecture for Foundation Model-Driven Speech Anti-Spoofing," IEEE Transactions on Information Forensics and Security, 2025\.

\[14\] D. Combei, A. Stan, D. Oneata, and H. Cucu, "WavLM Model Ensemble for Audio Deepfake Detection," in Proc. ASVspoof Workshop, 2024\. arXiv:2408.07414.


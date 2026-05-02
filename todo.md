TODO List

[x] Read the main survey paper (Li et al., 2024) and the PMC survey (Zhang et al., 2025) to build a solid overview of the field
[x] Read AASIST, WavLM, and Wav2Vec2+AASIST papers to understand the SOTA architectures
[x] Set up environment and pre-trained models, download ASVspoof 2019 LA dataset.
    [x] Run AASIST (pretrained) on ASVspoof 2019 LA eval and record EER. 
    [x] Run Nes2Net (WavLM, pretrained) on ASVspoof 2019 LA eval and record EER. 
    [x] Run AASIST3 (HuggingFace, pretrained) on ASVspoof 2019 LA eval and record EER. 
    [x] Run ASVspoof 2021 baseline (LFCC+LCNN, pretrained) on ASVspoof 2019 LA eval and record EER
[x] Re-run the above models on ASVspoof 2021 DF eval (codec-heavy) and/or In-the-Wild dataset
[] [in-progress] Error analysis - understand each model's capability on each dataset
    [] Setup: load all result pkl + protocol metadata, build unified per-dataset DataFrame
       with utt_id, label, metadata, model scores, predictions, correctness
       (label convention: 1=bonafide, 0=spoof)
    [] Per-dataset capability analysis: EER, threshold at EER, FAR/FRR,
       score distributions, calibration signal, confident errors
    [] Group errors by available metadata
       (ASV19 attack; ASV21 codec/vocoder/attack; ASV5 attack/codec/condition;
        In-the-Wild speaker/duration if available)
    [] Hard-error mining: confident false positives/false negatives and
       representative spectrograms when audio is available
    [] Cross-model analysis: score correlation, unique/shared errors,
       universal hard examples
    [] Cross-dataset synthesis: EER table, generalization gap vs ASV19,
       degradation patterns, per-model strengths/weaknesses, improvement hypotheses
[] Evaluate on additional datasets for broader coverage
    [x] In-The-Wild (2022)
    [x] ASVspoof 5 (2024) — 32 attacks, adversarial, multi-codec (Part 8)
    [] EchoFake (2025) — physical replay attacks + modern TTS (Part 7)
[] Propose a more robust approach based on observed weaknesses (e.g., codec augmentation, SSL fine-tuning strategy, ensemble)
        [] Implement and quickly evaluate the proposed improvement
[] Prepare final presentation slides

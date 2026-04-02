TODO List
[] Read the main survey paper (Li et al., 2024) and the PMC survey (Zhang et al., 2025) to build a solid overview of the field
[] Read AASIST, WavLM, and Wav2Vec2+AASIST papers to understand the SOTA architectures
[] Set up environment and pre-trained models, download ASVspoof 2019 LA dataset.
    [] Run AASIST (pretrained) on ASVspoof 2019 LA eval and record EER. 
    [] Run Nes2Net (WavLM, pretrained) on ASVspoof 2019 LA eval and record EER. 

    [] Run AASIST3 (HuggingFace, pretrained) on ASVspoof 2019 LA eval and record EER. 
    [] Run ASVspoof 2021 baseline (LFCC+LCNN, pretrained) on ASVspoof 2019 LA eval and record EER
[] [in-progress] Re-run the above models on ASVspoof 2021 DF eval (codec-heavy) and/or In-the-Wild dataset
[] [in-progress] Compare results across models and datasets; identify where each model succeeds and fails
    [] Synthesize insights on pros/cons of each approach (hand-crafted vs end-to-end vs SSL-based)
[] Propose a more robust approach based on observed weaknesses (e.g., codec augmentation, SSL fine-tuning strategy, ensemble)
        [] Implement and quickly evaluate the proposed improvement
[] Prepare final presentation slides

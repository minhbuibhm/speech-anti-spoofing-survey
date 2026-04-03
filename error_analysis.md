# Error Analysis Plan

## Abstract

Beyond overall EER, we need fine-grained analysis to understand *where* and *why* each model fails. The core idea: re-use existing per-utterance scores (no re-inference needed) and enrich them with metadata from protocol files (attack type, codec, vocoder, speaker). This enables group-level EER breakdowns that reveal model-specific weaknesses — the key insight for proposing improvements.

Three priorities: (1) quantitative breakdowns by attack/codec/vocoder, (2) cross-model complementarity analysis, (3) targeted listening via stratified sampling.

## Available Data

| Dataset | Models | Utterances | Stored in |
|---------|--------|------------|-----------|
| ASVspoof 2019 LA | AASIST, AASIST-L, AASIST3, LFCC+LCNN | 71,237 | `ALL_RESULTS` / `.pkl` |
| ASVspoof 2021 DF | AASIST, LFCC+LCNN | ~611K | `RESULTS_2021` |

Current limitation: only `scores` and `labels` are saved. Metadata (attack type, codec, speaker, vocoder) exists in protocol files but is not yet extracted.

## Recommended Analyses

### Priority 1 — Grouped EER Breakdowns

Parse protocol files to attach metadata to each utterance, then compute EER per group. No re-inference required.

| Analysis | Grouping field | Protocol index | Dataset | Key question |
|----------|---------------|----------------|---------|-------------|
| EER by attack type | A01–A19 | 2019: `parts[3]`, 2021: `parts[4]` | Both | Which TTS/VC attacks are hardest? |
| EER by codec | nocodec, low_m4a, mp3m4a… | 2021: `parts[2]` | 2021 DF | Which codecs cause most degradation? |
| EER by vocoder type | traditional / neural_auto / neural_nonauto | 2021: `parts[8]` | 2021 DF | Neural vocoders harder than traditional? |
| EER by speaker | Speaker ID | `parts[0]` | Both | High variance across speakers? |

**Expected outcome:** tables + bar charts showing per-group EER for each model side by side.

### Priority 2 — Cross-Model Complementarity

Using existing scores from both models on the same utterances:

- **Score correlation**: scatter plot of AASIST vs LFCC+LCNN scores per utterance. Low correlation → ensemble potential.
- **Failure overlap**: classify each utterance as correct/incorrect per model (using EER threshold), then count overlap. If models fail on different utterances → strong ensemble signal.

**Expected outcome:** correlation coefficient + Venn diagram of failure sets.

### Priority 3 — Stratified Listening

Goal: listen to a small representative sample (~30–50 utterances) that covers the full distribution of failure modes.

Sampling strategy:
1. From Priority 1 results, identify the top failure groups (e.g., worst 3 attack types × worst 2 codecs).
2. Within each group, pick 2–3 utterances: one confident false positive, one confident false negative, one near-threshold.
3. Optionally compare with spectrogram plots (mel-spec of bonafide vs missed spoof).

### Optional — Score Shift Analysis (2019 → 2021)

For attack types present in both datasets (e.g., A07–A19), compare score distributions with and without codec compression. Quantifies exactly how much codec degrades each model's confidence.

## Next Steps

1. Write a cell that parses protocol files and builds a DataFrame with columns: `utt_id, score_aasist, score_lfcc, label, attack, codec, vocoder, speaker`.
2. Run Priority 1 analyses, generate tables and plots.
3. Run Priority 2, assess ensemble potential.
4. Select samples for Priority 3 based on results.

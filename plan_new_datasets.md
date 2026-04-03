# Plan: Evaluate on EchoFake & ASVspoof 5

## Goal

Run AASIST and LFCC+LCNN on two new datasets to test generalization beyond ASVspoof 2019/2021. Report overall EER per model per dataset — same format as existing results.

## Part 7 — EchoFake (2025)

**Why:** First dataset with real physical replay attacks. ASVspoof 2019 LA / 2021 DF only have digital injection (TTS/VC) — no physical replay.

**Source:** `EchoFake/EchoFake` on HuggingFace

**Dataset summary:**
- 4 labels: bonafide (B), replayed bonafide (RB), fake (F), replayed fake (RF)
- Audio: 16 kHz mono, MP3 64 kbps
- Eval splits: `closed_set_eval` (6K samples), `open_set_eval` (25.6K samples)
- TTS models: XTTSv2, F5-TTS, SpeechT5, LLaSA-1B, OpenAudio-S1, StyleTTS2 (closed), CosyVoice2, IndexTTS, MaskGCT, OpenVoice-V2, FireRedTTS-1 (open)
- Replay conditions: 2 rooms, 2 playback devices, 2 recorders, 2 distances (closed); unseen room/devices (open)

**Evaluation approach:**
- Binary EER: bonafide (B only) vs spoof (RB + F + RF)
- Also compute EER per label category: B vs RB, B vs F, B vs RF — to see which attack type each model struggles with

**Loading data:**
```python
from datasets import load_dataset

ds = load_dataset("EchoFake/EchoFake")
eval_closed = ds['closed_set_eval']
eval_open = ds['open_set_eval']

# Each sample: sample['path'] (audio), sample['label'] (str), sample['synthesis_details'], sample['replay_details']
# Map labels: bonafide → 1, {replayed_bonafide, fake, replayed_fake} → 0
```

**Steps:**
1. `pip install datasets` (should already be on Kaggle)
2. Load `closed_set_eval` and `open_set_eval` splits
3. Map 4-class labels to binary (bonafide=1, rest=0)
4. Audio is already 16 kHz — load via `sample['path']` (HF datasets handles decoding)
5. Run AASIST inference → scores + labels → compute EER
6. Run LFCC+LCNN inference → scores + labels → compute EER
7. Also compute per-category EER (B vs RB, B vs F, B vs RF)
8. Save results to `RESULTS_ECHOFAKE` dict, same format as `ALL_RESULTS`

**Estimated time:** ~6K + 25.6K = ~31.6K utterances. At ~50 utt/s (AASIST) = ~10 min per model.

**Gotchas:**
- Audio is MP3, not FLAC — HF datasets decodes to numpy array, need to convert to tensor
- HF audio column returns `{'array': np.array, 'sampling_rate': int}` — use `sample['path']['array']`
- May need `trust_remote_code=True` if dataset uses custom loading script

**Progress:**
- [x] Cells added to `survey.ipynb` (7.0–7.4)
- [x] 7.0 — Load dataset from HF, print label distribution per split
- [x] 7.1 — Helper functions: `echofake_to_wav_tensor` (HF audio → tensor), `predict_aasist_wav`, `predict_lcnn_wav` (in-memory inference, no file I/O)
- [x] 7.2 — AASIST eval on both `closed_set_eval` + `open_set_eval`, stores `categories` for per-label breakdown
- [x] 7.3 — LFCC+LCNN eval on both splits
- [x] 7.4 — Per-category EER table (B vs RB / F / RF), cross-dataset summary (ASV19 + ASV21 + EchoFake), save `.npz`
- [ ] Run on Kaggle — verify label names match (may need to adjust `'replay'`/`'fake'`/`'replayed_fake'` in 7.4 after seeing actual label strings from 7.0)
- [ ] Check if HF audio column name is `'path'` or `'audio'` — adjust `echofake_to_wav_tensor` if needed

---

## Part 8 — ASVspoof 5 (2024)

**Why:** 32 attack types (vs 13 in 2019), includes adversarial attacks (Malafide/Malacopula), multi-codec (Opus, AMR, Speex, EnCodec, mp3, m4a). Much harder generalization test.

**Source:** `jungjee/asvspoof5` on HuggingFace

**Dataset summary:**
- Binary labels: bonafide vs spoof
- 32 attacks: A01–A08 (train), A09–A16 (dev), A17–A32 (eval) — includes adversarial
- Codecs: C00 (none), Opus, AMR, Speex, EnCodec, mp3, m4a at 8/16 kHz
- Eval Track 1: ~138K bonafide + ~542K spoof = ~680K utterances (very large)
- Audio: FLAC format

**Evaluation approach:**
- Use `eval.track_1.tsv` protocol
- Binary EER (bonafide vs spoof), same as ASVspoof 2019/2021
- If 680K is too slow, subsample eval set or use dev set (~140K, still large)

**Loading data:**
```python
# Option A: HuggingFace datasets (if supported)
ds = load_dataset("jungjee/asvspoof5")

# Option B: Manual download (more likely needed — dataset is tar files)
# Download flac_D_*.tar, extract, use protocol TSV to iterate
```

**Protocol format** (TSV, from dataset page):
```
speaker_id  utt_id  gender  -  -  codec  attack  label  ...
D_0062      D_0000000001  F  -  -  AC1  A11  spoof  -
```

**Steps:**
1. Check if HF datasets loader works, otherwise download tar + extract
2. Parse `eval.track_1.tsv` (or `dev.track_1.tsv` if eval too large)
3. Load audio by utt_id from `flac_D/` directory
4. Run AASIST inference → compute EER
5. Run LFCC+LCNN inference → compute EER
6. Save results

**Estimated time:**
- Dev set (~140K): ~45 min per model at 50 utt/s
- Eval set (~680K): ~3.5 hours per model — probably too long for Kaggle session
- **Recommendation:** Use dev set first, or subsample eval (e.g., 50K random utterances)

**Gotchas:**
- Dataset is very large (~100+ GB for eval audio). Kaggle disk limit is ~70 GB.
- May need to stream or process in chunks without fully downloading.
- HF dataset may require `streaming=True` mode.
- Protocol file is TSV not space-separated — use `pd.read_csv(..., sep='\t')`

---

## Priority & Order

1. **EchoFake first** — smaller (~31K eval), easy HF loading, novel replay insight
2. **ASVspoof 5 dev set second** — ~140K, manageable on Kaggle, 16 new attack types
3. ASVspoof 5 eval set only if time permits (streaming mode or subsample)

## Output Format

Same as existing results — dict with `{model_name: {'eer': float, 'scores': np.array, 'labels': np.array}}`, saved to `.npz` for reuse in Part 6 error analysis.

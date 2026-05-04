# Evaluation Notebooks Summary

This folder contains four standalone Kaggle notebooks for dataset-level evaluation plus one CPU-only error-analysis notebook.

## Shared result format

Each notebook writes a unified `results.pkl`:

```python
{
    "__metadata__": {  # optional, added by error_analysis backfill
        "dataset": str,
        "utt_ids": np.ndarray,
        "label_convention": "1=bonafide,0=spoof",
    },
    "AASIST": {"eer": float, "scores": np.ndarray, "labels": np.ndarray},
    ...
}
```

`labels` use `1=bonafide` and `0=spoof`. `scores` are always bonafide probabilities, so EER uses bonafide as the positive class.
The optional `__metadata__` entry is dataset-level metadata; evaluation and analysis code skip it when iterating model results.

## Resume behavior

Each model writes `<model>.partial.npz` during inference with `scores`, `labels`, and `utt_ids`. When a run is restarted, the notebook skips completed utterance IDs. Once a model finishes, the notebook immediately updates the dataset-level `results.pkl` and removes that model's partial file.

## ASVspoof 2019 LA

Notebook: `eval_asvspoof_2019.ipynb`

Expected Kaggle input paths:

- `/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset`
- fallback `/kaggle/input/asvpoof-2019-dataset`

The LA CM protocol row is `speaker utt_id - attack key`. Evaluation uses `ASVspoof2019_LA_eval/flac` and `ASVspoof2019.LA.cm.eval.trl.txt`. Existing `sdd-survey/asvspoof19/results.pkl` is reused, but AASIST3 is forced by default because the old score used the wrong class index.

## ASVspoof 2021 DF

Notebook: `eval_asvspoof_2021.ipynb`

Expected Kaggle input paths:

- `/kaggle/input/datasets/mohammedabdeldayem/avsspoof-2021`
- fallback `/kaggle/input/avsspoof-2021`

DF eval audio is split across `ASVspoof2021_DF_eval_part00/01/02`. The CM key is field index 5 in `trial_metadata.txt`. Existing AASIST and LFCC+LCNN results are reused unless forced; missing AASIST3 and AASIST-L entries are evaluated and appended.

## ASVspoof 5 Track 1

Notebook: `eval_asvspoof_5.ipynb`

This notebook evaluates Track 1 only: stand-alone bonafide/spoof countermeasure scoring. Track 2 files are for SASV enrollment/trials and are not compatible with the current `results.pkl` CM score format.

Important ASVspoof 5 structure:

- `ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`, `ASVspoof5.eval.track_1.tsv` are space-separated Track 1 metadata.
- Track 1 columns are `SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`.
- `KEY` is the binary CM label: `bonafide` -> 1, `spoof` -> 0.
- Audio prefixes are `flac_T` for train, `flac_D` for dev, and `flac_E` for eval.
- The Hugging Face mirror `jungjee/asvspoof5` is a WebDataset-style package with protocol files and large FLAC tar shards. `eval_asvspoof_5.ipynb` defaults to `ASV5_SOURCE='auto'`: it uses local extracted files if present, otherwise falls back to Hugging Face streaming. Kaggle Internet must be enabled for this fallback.
- In HF mode, the notebook now prints `inspect_asv5_hf_stream()` output and builds `ASV5_HF_INDEX` before loading any model. If matched audio rows are zero, it stops early with the printed keys/fields instead of running inference and failing at EER computation.
- Because Hugging Face `datasets` does not expose ASVspoof 5 extensionless FLAC payloads reliably, the current default is `ASV5_SOURCE='hf_tar'`. It downloads `ASVspoof5_protocols.tar` and then processes the official eval shards one at a time (`flac_E_aa.tar` through `flac_E_aj.tar`): download one shard, run all enabled models on that shard, save partials, delete the shard, then continue to the next suffix.
- The generated ASVspoof 5 notebook now targets the official eval split (`ASVspoof5.eval.track_1.tsv`) by default. Older dev-split `results.pkl` files are ignored unless they explicitly declare matching dataset metadata. In HF-tar mode, the notebook also writes a copy of `ASVspoof5.eval.track_1.tsv` next to `/kaggle/working/asvspoof5/results.pkl` for upload back into the `sdd-survey` dataset.

## In-the-Wild

Notebook: `eval_in_the_wild.ipynb`

Müller et al. 2022 deepfake-in-the-wild dataset (~31K utterances) collected from social media. Standard cross-dataset generalization probe: none of the models were trained on it.

Expected Kaggle input paths:

- `/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake`
- fallback `/kaggle/input/in-the-wild-audio-deepfake`

The locator searches for `meta.csv` (columns `file`, `speaker`, `label`) and a flat audio directory. `label='bona-fide'` maps to 1, anything else to 0.

## Error Analysis

Notebook: `error_analysis.ipynb`

This notebook consumes saved `results.pkl` files and lightweight protocol metadata only. When a pickle lacks evaluated `utt_ids`, it reconstructs the eval index without inference, writes patched pickles under `/kaggle/working/results_with_utt_ids/<dataset>/results.pkl`, and uses `utt_id` joins for metadata alignment. ASVspoof 2021 backfill mirrors the eval notebook's narrow audio-directory scan and checks audio loadability to reproduce the eval loader's skip-on-decode-error behavior. ASVspoof 5 now prioritizes the official eval Track 1 protocol; the eval notebook writes dataset-level `__metadata__.utt_ids`, so normal error analysis no longer needs to download ASVspoof 5 tar shards. It writes deterministic artifacts under `/kaggle/working/error_analysis/`, including per-dataset `error_analysis.pkl`, metrics CSVs, grouped EER CSVs, score-distribution plots, failure-overlap tables, hard-error tables, a cross-dataset synthesis folder, and `error_analysis_artifacts.zip` containing both `error_analysis/` and `results_with_utt_ids/` for upload back to the `sdd-survey` Kaggle dataset.

## Model score convention

AASIST, AASIST-L, AASIST3, XLS-R+AASIST, XLS-R+Nes2Net all produce two-class logits ordered as `[spoof, bonafide]` in these notebooks. The saved score is therefore:

```python
torch.softmax(logits, dim=1)[:, 1]
```

This is especially important for AASIST3. Using index 0 inverts the bonafide score and causes the ASVspoof 2019 EER to look wrong.

## SSL model additions: research findings, decisions, and code changes

This section documents what changed when the two SSL-based models (`XLS-R+AASIST`, `XLS-R+Nes2Net`) and the In-the-Wild notebook were added in the 2026-04 update.

### Research findings (web search, 2026-04)

The original survey only covered hand-crafted (`LFCC+LCNN`) and end-to-end graph-attention models (`AASIST`, `AASIST-L`, `AASIST3`). According to the broader literature (Tak et al. 2022, Stourbe et al. 2024, Liu et al. 2025) the dominant SOTA paradigm is now SSL front-end + lightweight back-end:

- WavLM-Large or XLS-R as the SSL front-end (each ~300M parameters).
- A small back-end (AASIST graph attention, MHFA, or Nes2Net nested Res2Net).
- WavLM is best on clean ASVspoof 2019 LA, XLS-R is reported best on ASVspoof DF, HuBERT is best on In-the-Wild.

In-the-Wild itself was already on `todo.md` as a generalization probe but had not been wired into a notebook. ASVspoof 2021 DF, ASVspoof 5, and ASVspoof 2019 LA covered codec compression and standard CM testing but not real-world social-media audio.

### Decisions

1. **Two new models** are added to the survey:
   - `XLS-R+AASIST` — Tak et al. 2022 (Odyssey), repo `TakHemlata/SSL_Anti-spoofing`, weights `Best_LA_model_for_DF.pth`.
   - `XLS-R+Nes2Net` — Liu et al. 2025, repo `Liu-Tianchi/Nes2Net_ASVspoof_ITW`, model class `wav2vec2_Nes2Net_no_Res_w_allT`. Reports 1.49% EER on ASVspoof 2021 DF and 5.52% EER on In-the-Wild.

   Both share the same XLS-R 300M front-end so the comparison isolates the back-end (AASIST graph attention vs Nes2Net nested Res2Net), which is one direct comparison the survey wanted.

   `WavLM+Nes2Net` and `WavLM+AASIST` were considered but rejected:
   - The `Liu-Tianchi/Nes2Net` main repo only ships `WavLM_Nes2Net*` checkpoints trained on the CtrSVDD singing-voice deepfake set, which is the wrong domain for ASVspoof / In-the-Wild evaluation.
   - No widely-released `WavLM+AASIST` pretrained weights for speech anti-spoofing exist.

2. **One new dataset notebook**: In-the-Wild (Kaggle slug `abdallamohamed312/in-the-wild-audio-deepfake`).

3. **Compound runner (option Z)** is available but disabled by default. When enabled, several models that share the same dataset class run in one pass over the data loader so each utterance is decoded only once. It stays opt-in because loading two XLS-R 300M models simultaneously can exceed Kaggle T4/P100 memory. The XLS-R front-ends inside the two SSL models are NOT shared because each pretrained checkpoint contains independently fine-tuned SSL weights (TakHemlata trained end-to-end; Liu-Tianchi used `lr=2.5e-7` which is consistent with SSL fine-tuning, and the README averaged checkpoint name encodes that learning rate). Sharing forward output would silently feed the wrong SSL features into the second back-end. The realised win is therefore audio I/O, not front-end compute.

   Per-batch audio caching (option Y) was rejected because XLS-R features at float16 are ~414 KB per utterance, so caching ASVspoof 2021 DF (~190 GB) and ASVspoof 5 eval (~280 GB) exceeds the 70 GB Kaggle working disk.

### Code changes in `scripts/create_eval_notebooks.py`

- `MODEL_CODE`:
  - Added `ensure_ssl_aasist_repo`, `ensure_nes2net_repo`, `ensure_fairseq_source`, `find_or_download_xlsr_300m`, `_link_xlsr_into`, `find_xlsr_aasist_checkpoint`, `find_xlsr_nes2net_checkpoint`, `_strip_state_prefix`, `_reset_module_namespace`, `load_xlsr_aasist_model`, `load_xlsr_nes2net_model`, `predict_ssl_pair_batch`.
  - Registered `XLS-R+AASIST` and `XLS-R+Nes2Net` in `MODEL_REGISTRY` (both with `WaveformDataset`, batch size 8).
  - Added `COMPOUND_GROUPS` declaring `[XLS-R+AASIST, XLS-R+Nes2Net]` as a shared-loader group.
  - Added safety checks: fairseq is installed from the pinned XLS-R-compatible source snapshot instead of PyPI; SSL checkpoints must load with no missing/unexpected keys; generic module namespaces are reset before switching between cloned repos.
  - Nes2Net args follow the repo CLI defaults: `n_output_logits=2`, `dilation=2`, `pool_func='mean'`, `SE_ratio=[1]`, `Nes_ratio=[8, 8]`.
  - `predict_ssl_pair_batch` now expects a single `[batch, 2]` logits tensor and raises immediately on any unexpected output shape.

- `EVAL_LOCAL_CODE`:
  - Added `evaluate_models_compound` that loads multiple models simultaneously, iterates the loader once, runs each model on every batch, and saves each model's partial / pickle independently.
  - Added `run_eval_plan` that schedules compound and single-model runs based on `ENABLED_MODELS`, `COMPOUND_GROUPS`, and `RUN_COMPOUND_SSL_MODELS`. The default is sequential execution for VRAM safety.

- `RUN_LOCAL_CODE` and the `local` branch of `RUN_ASV5_CODE` switched to call `run_eval_plan(...)` instead of looping `evaluate_model_on_dataframe`.

- `ASV2019_DATASET_CODE`, `ASV2021_DATASET_CODE`, `ASV5_DATASET_CODE` now include `XLS-R+AASIST` and `XLS-R+Nes2Net` in `ENABLED_MODELS` and `FORCE_EVAL`.

- New constants `IN_THE_WILD_DATASET_CODE` (locator + parser for `meta.csv` schema) and `build_in_the_wild()` produce a fourth notebook `eval_in_the_wild.ipynb`. The parser builds a recursive audio index by relative path, filename, and stem to tolerate Kaggle layout differences. `main()` writes it alongside the existing three.

### Review fixes after adding SSL models

The first implementation was intentionally reviewed before running full experiments. The following issues were fixed:

- Partial checkpoint loading for SSL models was removed. Missing or unexpected checkpoint keys now raise an error, because partial loads can silently produce invalid EER values.
- The fairseq dependency now uses each repo's pinned `fairseq-*` folder when present; otherwise it clones `pytorch/fairseq` at commit `a54021305d6b3c4c5959ac9395135f63202db8f1` into `/kaggle/working`. PyPI fairseq is intentionally avoided because Kaggle's current pip rejects old `omegaconf` metadata and the released wheel can drift from the XLS-R checkpoint code.
- The fairseq bootstrap downgrades to `pip<24.1`, installs Python-3.12-compatible runtime/build dependencies, and imports fairseq directly from the patched source tree on `sys.path`. It intentionally avoids editable fairseq builds, because the old fairseq packaging path is brittle on Kaggle's Python 3.12 runtime.
- The pinned fairseq source directory is inserted into `sys.path`, and the notebook verifies `import fairseq` immediately after dependency setup. This keeps the active kernel pointed at the patched source tree.
- Old fairseq dataclass config defaults are patched automatically for Python 3.12 by rewriting mutable defaults like `common: CommonConfig = CommonConfig()` to `field(default_factory=CommonConfig)`.
- Hydra 1.0.x is installed as a fairseq dependency and has the same Python 3.12 dataclass issue. The notebook patches `hydra/conf/__init__.py` before importing fairseq, then clears `fairseq`, `hydra`, and `omegaconf` from `sys.modules` for a clean import.
- Because fairseq's old `hydra_init()` reads dataclass `.default`, it also needs a small patch after converting fields to `default_factory`; the notebook now instantiates `field_info.default_factory()` before registering configs with Hydra.
- The notebook prints `fairseq.__file__` after import so fairseq namespace conflicts between SSL repos are visible. SSL models still run sequentially by default, which is the safer path for both VRAM and fairseq import stability.
- Compound XLS-R evaluation is disabled by default through `RUN_COMPOUND_SSL_MODELS=False` to avoid T4/P100 out-of-memory failures.
- Generic Python module namespaces (`model`, `models`, `model_scripts`) are cleared before switching cloned repos to avoid stale imports.
- In-the-Wild audio discovery now uses a recursive audio index instead of assuming all files are directly under one flat directory.

### Required Kaggle input paths for the SSL models

- `/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr_aasist/Best_LA_model_for_DF.pth` — TakHemlata pretrained anti-spoofing weights (Google Drive link in their README).
- `/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/wav2vec2_nes2net/<file>.pth` — Liu-Tianchi pretrained averaged checkpoint, e.g. `wav2vec2_Nes2Net_X_e100_bz12_lr2.5e_07_algo4_seed12345_avg_ckpt_ep59_61_69.pth`. If it is not attached as a Kaggle dataset, the notebook tries to download Google Drive id `1JFGv_2TONMnTLGbiOIuHFfMvuo4SIIpg` to `/kaggle/working/wav2vec2_nes2net/pretrained_nes2net.pth`.
- `/kaggle/input/datasets/minhbhm/sdd-survey/checkpoints/xlsr/xlsr2_300m.pt` — XLS-R 300M base SSL checkpoint. The notebooks now preflight this Kaggle input before running SSL models, so attach it as a dataset instead of relying on runtime download. Direct URL: `https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt`.

### References used for dataset structure

- ASVspoof 5 Hugging Face dataset card: https://huggingface.co/datasets/jungjee/asvspoof5
- ASVspoof 5 README on Hugging Face, including Track 1 metadata columns and FLAC counts: https://huggingface.co/datasets/jungjee/asvspoof5/blob/main/README.txt
- TakHemlata/SSL_Anti-spoofing (XLS-R+AASIST baseline, weights via Google Drive): https://github.com/TakHemlata/SSL_Anti-spoofing
- Liu-Tianchi/Nes2Net_ASVspoof_ITW (XLS-R+Nes2Net-X, weights via Google Drive): https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW
- Nes2Net paper (arXiv 2504.05657): https://arxiv.org/abs/2504.05657
- In-the-Wild paper / dataset (Müller et al. 2022): https://deepfake-total.com/in_the_wild

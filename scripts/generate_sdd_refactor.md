# SDD Refactor Notes

## Current State

The source of truth for the generated refactored notebook is:

- `scripts/generate_sdd_refactor.py`
- generated notebook: `sdd-refactor.ipynb`
- original exploratory notebook: `sdd-review.ipynb`

The refactor target is still the same: a simple dataset-first benchmark notebook that can reuse saved artifacts, evaluate missing results only when necessary, and run comprehensive error analysis after each dataset.

The intended dataset order is:

1. `ASVspoof 2019 LA`
2. `ASVspoof 2021 DF`
3. `ASVspoof 5`

The intended model set is:

- `AASIST`
- `AASIST-L`
- `AASIST3`
- `LFCC+LCNN`

## Verified State From `sdd-review.ipynb`

The old notebook already contains usable saved results for ASVspoof 2019 and ASVspoof 2021 DF. These are important because they avoid expensive reruns.

### ASVspoof 2019 LA

Observed artifact:

- Kaggle path used by the old notebook:
  `/kaggle/input/datasets/minhbhm/asvspoof19-results/all_results_asvspooft_19.pkl`

Observed format:

- one pickle file
- top-level dictionary keyed by model name
- each model entry contains:
  - `eer`
  - `scores`
  - `labels`

Observed saved models:

- `AASIST`: `4.59%` EER
- `AASIST-L`: `6.74%` EER
- `LFCC+LCNN`: `19.64%` EER
- `AASIST3`: `79.19%` EER

Observed alignment state:

- each model has `71,237` scores
- this matches the ASVspoof 2019 LA eval protocol size
- the old notebook attached scores directly in protocol order
- the old artifact does not include `utt_id`

### ASVspoof 2021 DF

Observed artifact files:

- `scores_2021DF_AASIST.npz`
- `scores_2021DF_LFCC_LCNN.npz`

Observed format:

- one `.npz` file per model
- each file contains:
  - `scores`
  - `labels`
  - `eer`

Observed saved models:

- `AASIST`: `17.70%` EER
- `LFCC+LCNN`: `33.81%` EER

Observed alignment state:

- protocol rows in old notebook: `611,829`
- existing audio files found: `458,871`
- `AASIST` saved scores: `458,868`
- `LFCC+LCNN` saved scores: `458,868`
- old notebook tolerated the small `3` row mismatch as decode-error fallout
- the old artifact does not include `utt_id`

## Dataset Reality And References

The refactor should match the real protocol structure, not only the local notebook assumptions.

### ASVspoof 2019 LA

ASVspoof 2019 LA is a logical-access spoofing dataset with train, development, and evaluation splits. The eval split contains:

- `7,355` bonafide utterances
- `63,882` spoof utterances
- `71,237` utterances total

The protocol provides speaker ID, utterance ID, system/attack ID, and bonafide/spoof key. Bonafide rows use `-` for the attack/system field.

References:

- Edinburgh DataShare ASVspoof 2019 dataset page: https://datashare.ed.ac.uk/handle/10283/3336?show=full
- Hugging Face ASVspoof 2019 dataset card: https://huggingface.co/datasets/LanceaKing/asvspoof2019

### ASVspoof 2021 DF

ASVspoof 2021 DF is the deepfake track. Its metadata includes more analysis fields than ASVspoof 2019:

- speaker ID
- trial ID
- codec
- source database
- spoofing attack
- bonafide/spoof key
- trim flag
- subset
- vocoder category

The official evaluation package describes the DF metadata format with this example:

`LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -`

The DF track uses EER as the relevant metric.

References:

- ASVspoof 2021 official page: https://www.asvspoof.org/index2021.html
- ASVspoof 2021 official eval package README: https://github.com/asvspoof-challenge/2021/blob/main/eval-package/README.md
- ASVspoof 2021 DF Zenodo page: https://zenodo.org/records/4835108

### ASVspoof 5

ASVspoof 5 is less stable operationally in the current notebook because it may come from either mounted local files or Hugging Face fallback. Those two modes must not be mixed accidentally when using cached indexes.

Reference:

- ASVspoof 5 evaluation plan: https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf

## Main Problems Found

### 1. Error Analysis Could Break On Missing Common Columns

The previous generated notebook assumed every dataset metadata table had the same columns, including `hf_index`.

That was not true:

- ASVspoof 2019 metadata did not include `hf_index`
- ASVspoof 2021 metadata did not include `hf_index`
- ASVspoof 5 needs `hf_index` only in Hugging Face mode

This could cause error analysis to fail before any grouped EER tables were produced.

### 2. The Pipeline Was Not Truly Artifact-First

The previous generated notebook loaded all dataset indexes and all enabled models early.

That means a run intended to reuse cached ASVspoof 2019 scores could still fail because:

- ASVspoof 2021 paths were unavailable
- ASVspoof 5 Hugging Face fallback failed
- AASIST/AASIST3 repositories or weights could not be downloaded

This violated the intended goal: reuse old 2019 and 2021 results whenever possible.

### 3. Legacy Score Alignment Was Too Implicit

The old 2019 and 2021 artifacts do not include `utt_id`, so they can only be reused by protocol order.

That is acceptable for compatibility, but it should not be silent. The saved `labels` should be checked against the metadata labels before attaching scores.

### 4. ASVspoof 5 Cached Indexes Could Mix Modes

ASVspoof 5 can be indexed from:

- local mounted files
- Hugging Face dataset rows

A cached Hugging Face-style index has no local `audio_path`. A cached local-file index has file paths that may not exist in a later session. The notebook needs to infer the cached mode and rebuild or reload the right handle.

## High-Level Plan

The simplest robust design is:

1. Standardize metadata shape for every dataset.
2. Load/build one dataset at a time.
3. Check saved score artifacts before loading any model.
4. Use legacy order-based alignment only after label validation.
5. Save comprehensive error analysis immediately after each dataset.
6. Let the cross-dataset summary work even when only some datasets have completed.

## Implementation In `generate_sdd_refactor.py`

The generator was updated and `sdd-refactor.ipynb` was regenerated.

### Metadata Standardization

Added a common metadata schema:

- `dataset`
- `split`
- `speaker`
- `utt_id`
- `source`
- `attack`
- `codec`
- `vocoder`
- `label`
- `audio_path`
- `hf_index`

All dataset parsers now return standardized metadata. Missing fields are filled with conservative defaults such as `unknown`, `nocodec`, or `None`.

### Dataset-Local Execution

Added lazy dataset loading:

- `get_dataset_bundle("asv19")`
- `get_dataset_bundle("asv21_df")`
- `get_dataset_bundle("asv5")`

Each dataset section now builds only the dataset it needs. This keeps ASVspoof 2019 reuse independent from ASVspoof 2021 and ASVspoof 5 availability.

### Cache-First Model Loading

Model registration is now lazy. The notebook registers model loaders, but does not immediately clone repositories or load weights.

For each dataset/model pair:

1. look for a saved result
2. validate alignment
3. reuse it if valid
4. only load/train the model if no valid saved result exists

This is especially important for ASVspoof 2019 and 2021 because the old results should be reused first.

### Legacy Artifact Reuse

ASVspoof 2019:

- still supports `all_results_asvspooft_19.pkl`
- supports both hyphen and underscore Kaggle dataset naming variants
- validates saved labels before attaching scores

ASVspoof 2021 DF:

- still supports `scores_2021DF_*.npz`
- maps `LFCC_LCNN` back to `LFCC+LCNN`
- supports old artifact dataset naming variants
- validates saved labels before attaching scores
- keeps small legacy order-based tolerance for decode-error mismatch

### Error Analysis

The error-analysis output remains comprehensive:

- `attached_scores.csv`
- grouped EER CSV/TXT/PNG tables
- failure overlap table
- score correlation plot where applicable
- per-dataset `analysis_summary.json`

Configured grouping:

- ASVspoof 2019: `attack`
- ASVspoof 2021 DF: `attack`, `codec`, `vocoder`
- ASVspoof 5: `attack`, `codec`, `vocoder` when available

### Cross-Dataset Summary

The cross-dataset summary now loads whatever dataset summaries exist. Missing dataset summaries are skipped instead of failing the whole summary cell.

This supports partial notebook runs.

## Validation Performed

Local structural checks completed:

- `scripts/generate_sdd_refactor.py` parses with `ast.parse`
- `python scripts/generate_sdd_refactor.py` regenerated `sdd-refactor.ipynb`
- generated notebook JSON loads successfully
- all non-magic generated notebook code cells parse successfully

Not yet completed:

- full Kaggle runtime execution
- actual artifact reuse test inside Kaggle
- full ASVspoof 5 local/Hugging Face execution

## Remaining Work

### 1. Validate In Kaggle

Run the notebook in Kaggle and confirm that the old artifacts are found:

- ASVspoof 2019 pickle
- ASVspoof 2021 DF per-model NPZ files
- LFCC checkpoint if available

### 2. Confirm Reuse Before Inference

For ASVspoof 2019:

- expected: all four old model results load without model inference
- expected: error analysis runs from cached scores

For ASVspoof 2021 DF:

- expected: `AASIST` and `LFCC+LCNN` load from NPZ
- expected: error analysis runs for attack, codec, and vocoder
- expected: missing models either run fresh or are skipped depending on enabled models and runtime constraints

### 3. Recheck AASIST3

`AASIST3` remains suspicious because the old 2019 EER was very poor (`79.19%`). Before trusting new AASIST3 results:

- run a small balanced sanity sample
- verify score direction
- confirm class index convention

### 4. Decide Long-Term Source Of Truth

For now, maintain the notebook through `scripts/generate_sdd_refactor.py`.

After runtime stabilization, decide whether:

- the generator remains the source of truth, or
- the notebook becomes the maintained artifact directly

## Practical Interpretation

The refactor is now closer to the real goal:

- ASVspoof 2019 and 2021 old results are prioritized for reuse.
- Dataset sections are more independent.
- Model loading is delayed until genuinely needed.
- Error analysis has a consistent metadata base and should be more comprehensive.

The next important step is a real Kaggle run, because path layout, attached artifact naming, and ASVspoof 5 mode selection can only be fully proven in that environment.

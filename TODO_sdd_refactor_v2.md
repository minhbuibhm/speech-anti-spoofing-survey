# TODO — `sdd-refactor-v2.ipynb` end-to-end readiness

## Goal

Make `sdd-refactor-v2.ipynb` produce a complete cross-model × cross-dataset comparison in a **single Kaggle session (≤ 12 h)**, with full crash-resume support so that progress is never lost between sessions.

- **Models (4):** `LFCC+LCNN`, `AASIST`, `AASIST-L`, `AASIST3`
- **Datasets (3):** ASVspoof 2019 LA eval, ASVspoof 2021 DF eval, ASVspoof 5 (split TBD — see §1)
- **Primary metric:** overall EER per (model, dataset), plus grouped EER by attack / codec / vocoder
- **Final deliverable:** one cross-dataset summary table + comparison plots, all artifacts saved under `/kaggle/working/sdd_refactor_results/` and uploaded back as a Kaggle dataset for the next session

## Goal properties (used to justify each TODO)

Every TODO below is tagged with which property of the goal it protects. If a TODO does not map to one of these, it is out of scope.

| Tag | Property | A failure looks like |
|-----|----------|----------------------|
| **C** | **Correctness** — the EER numbers in the final table must be valid | AASIST3 reports 79% on ASV19 because score is inverted; grouped EER mixes spoof rows with the wrong bonafide rows |
| **K** | **Completeness** — all 4 models × 3 datasets cells must be filled | ASV5 entries empty because indexer did not find the data; AASIST-L missing on ASV21 because cache miss never got re-run |
| **R** | **Resumability** — a dead session can continue without redoing finished work | Kaggle kernel times out at hour 11 with 80% of ASV21 done, next session restarts from zero |
| **T** | **Time-fit** — total runtime ≤ 12 h on Kaggle T4 | A single AASIST3 evaluation alone takes 4 h because inference is one utterance at a time |

## Success criteria

1. A cold start on Kaggle (no result datasets attached) completes all four models on all three datasets within 12 h using only pretrained / trained-once checkpoints.
2. A warm start (result dataset attached) skips all completed work and only fills missing entries.
3. A session that dies mid-inference can be resumed in the next session and only re-runs un-finished utterances.
4. The final cross-dataset table contains plausible EER for AASIST3 (≪ 79%); no silently inverted scores.
5. Per-dataset grouped EER (attack / codec / vocoder) is computed and saved.

---

## 1. Decisions to lock before implementation

These are blocking choices the user must confirm. Default recommendations are marked.

| # | Decision | Default recommendation | Why this default |
|---|----------|------------------------|------------------|
| D1 | ASVspoof 5 split | **`dev` (~140K utts)** | Fits in budget [T] with margin; covers 8 new attacks A09–A16 incl. adversarial → still demonstrates the comparison story without the 5 h cost of full eval |
| D2 | EchoFake inclusion | **Drop** for v2 scope | Current goal is 19/21/5 only; adding a 4th dataset adds ~1 h compute + indexer work [T] without strengthening the cross-codec story |
| D3 | Iteration order | **`for model: for dataset:`** | Each model loaded once → 3× fewer model loads [T]; only one model in GPU at a time → no co-residency OOM risk [K]; partial results still saved per (model, dataset) cell so progress is visible [R] |
| D4 | Inference batch size | AASIST family **64**, AASIST3 **8–16**, LFCC+LCNN **128** | AASIST3's Wav2Vec2+KAN backbone has the largest activations; smaller batch keeps it under T4 memory while still amortising kernel launch overhead [T] |
| D5 | LFCC+LCNN training | Train once on ASV19 train, reuse on all 3 datasets | Re-training per dataset would not match the cross-domain story we want to tell (we measure how a model trained on 19 generalises to 21/5); training once also saves ~30 min [T] |
| D6 | Score convention | All four models output `P(bonafide)` ∈ [0, 1]; labels `1 = bonafide, 0 = spoof`; EER computed with `pos_label=1` | Matches the official AASIST3 repo (`outputs[:, 1]`, `labels==1` is bonafide) and the legacy survey notebook → cached scores stay comparable [C] |

---

## 2. P0 — Blockers (the notebook will not produce correct results without these)

### P0.1 Fix `predict_aasist3` score index — tag: **C**

**Problem.** `sdd-refactor-v2.ipynb` Part 2 / cell 9 returns `probs[0]` (= `P(spoof)`) but `compute_eer` is called with `pos_label=1`. Score is therefore inverted and the cached AASIST3 EER on ASV19 is 79.19% instead of ~20.8%.

**Fix.**
```python
def predict_aasist3(model, audio_ref, device=DEVICE):
    """Return the bonafide score for the current AASIST3 checkpoint."""
    wav = prepare_aasist_input(preprocess_audio_ref(audio_ref)).to(device)
    with torch.no_grad():
        logits = model(wav)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[1])   # was probs[0]
```
Then invalidate the AASIST3 entry in `results/asvspoof19_results/all_results_asvspooft_19.pkl` and re-run.

**Why this matters for the goal.** The deliverable is a cross-model comparison. If one of the four models reports an EER that is wrong by 60 percentage points, the comparison is meaningless and any conclusion drawn from the table (e.g. "SSL-based models are most robust") is unsupported. This is a **correctness blocker** — every other improvement is wasted until this is fixed.

**Verification (against `mtuciru/AASIST3` repo).**
- `model/full_model.py`: `self.out_layer = KANLinear(5 * gat_dims[1], 2)` (2-logit head)
- `utils/validation.py:compute_scores`: `scores = outputs[:, 1]`
- `utils/metrics.py`: comment `"label of bona fide score is 1, label of spoof score is 0"`
- `utils/metrics.py:compute_antispoofing_metrics`: `genuine = scores[labels == 1]`

### P0.2 Fix Kaggle artifact mount paths — tag: **R, T**

**Problem.** Part 0 / cell 2 lists `ARTIFACT_MOUNT_ROOTS = [/kaggle/input, /kaggle/input/datasets/minhbhm, /kaggle/datasets/minhbhm]`. Kaggle actually mounts user datasets at `/kaggle/input/<slug>/...` — the `datasets/<user>/` prefix does not exist. As a result, every `find_existing_file` call returns `None` and every cache hit fails.

**Fix.**
1. Add a one-line diagnostic cell at the top of Part 0: `print(os.listdir('/kaggle/input'))` — this reveals the actual slugs after the user attaches their result datasets.
2. Replace `ARTIFACT_MOUNT_ROOTS` with `[Path('/kaggle/input'), DATASET_OUTPUT_DIRS[group]]`.
3. Replace `ARTIFACT_DATASET_NAMES` values with the real Kaggle slugs the user uploads (e.g. `sdd-refactor-results-asv19`).
4. Make `find_existing_file` print the resolved path the first time each group is hit (debug aid).

**Why this matters for the goal.** Without correct paths, every Kaggle session ignores the previous session's work and starts from zero. That single defect alone makes the 12-h budget blow up: a 7-h pipeline becomes effectively infinite because every session re-runs everything. This protects both **resumability** (cache must be discoverable) and the **time budget** (don't repeat finished work).

### P0.3 Implement batched inference — tag: **T**

**Problem.** Part 4 / cell 14 — `evaluate_one_model` iterates `eval_df.iterrows()` and calls `predict_fn(...)` per utterance. Each call runs the model on a batch of size 1, which leaves the T4 GPU mostly idle (Python loop overhead + kernel launch dominates).

**Fix.**
1. Wrap each model's `predict_fn` in a batched variant: `batch_predict(model, list[audio_ref]) -> np.ndarray[scores]`.
2. Use a `torch.utils.data.DataLoader` with a custom collate that pads/crops waveforms to the model's expected length (`cut=64600` for AASIST family; `max_frames=400` for LFCC).
3. Use the batch sizes from D4 (configurable via `RUN_CONFIG`).
4. On batch failure (rare audio decode error), re-run the offending batch one-by-one and skip the bad row — keeps the run going instead of aborting at hour 8.

**Why this matters for the goal.** With the current single-sample loop, AASIST3 alone on ASV21 DF (459K utts) is roughly 4 h, and the same model on ASV5 dev adds another 1 h. The combined pipeline cannot finish in a 12-h session. Batched inference is conservatively a 5–10× speedup on AASIST3 — it's the **single largest factor that decides whether the goal fits in 12 h**.

### P0.4 Release model GPU memory between models — tag: **K, T**

**Problem.** `MODEL_REGISTRY[name]["model"]` keeps every loaded model resident for the whole session. Under the current `for dataset: for model:` order (or once we switch to `for model: for dataset:`), the registry can hold AASIST + AASIST-L + AASIST3 (~300 M params, Wav2Vec2 included) at the same time, which OOMs on a 16 GB T4.

**Fix.** After a model finishes its scheduled work for the session (under D3, that means after all 3 datasets are evaluated for that model), set `model_entry["model"] = None` and call the existing `release_model(...)` helper. Verify with `torch.cuda.memory_allocated()` that VRAM drops back to baseline before loading the next model.

**Why this matters for the goal.** OOM mid-session = restart = wasted hours = goal not delivered. Even when not OOM, holding all three AASIST variants resident inflates the working set, slows allocator behaviour, and increases the risk of cascading failures. This protects **completeness** (no aborted runs) and gives back enough headroom to push AASIST3 batch size higher → indirectly **time-fit**.

### P0.5 ASVspoof 5 indexer (locator + protocol parser) — tag: **C, K**

**Problem.** Part 1 / cells 5–7 — `locate_asvspoof5_local` uses very loose globs (`*flac*{split}*`) that can match unrelated folders, and `index_asvspoof5_local` assumes column ordering `{0:speaker, 1:utt_id, 5:codec, 6:attack, 7:label}` without verifying against the actual ASV5 TSV header. The HuggingFace fallback `jungjee/asvspoof5` is also not viable for full eval (~100 GB FLAC vs. 70 GB Kaggle disk).

**Fix.**
1. Replace the glob locator with an explicit Kaggle slug lookup (e.g. `/kaggle/input/asvspoof5-dev/...` once the user confirms the slug).
2. Read the TSV with `pd.read_csv(sep='\t')`. If the file has a header row, use named columns; otherwise document and assert the column order from `plan_new_datasets.md`: `speaker_id, utt_id, gender, -, -, codec, attack, label, ...`.
3. Drop the HF fallback for the dev path (per D1 the dev split fits locally).
4. After parse, print label distribution + attack/codec value counts + sample audio path existence, so any silent indexing error is caught immediately.

**Why this matters for the goal.** ASVspoof 5 is one third of the cross-dataset comparison. If the indexer either points at the wrong files or maps columns incorrectly, the entire ASV5 column in the final table is junk — the comparison is missing one of its three legs. This is both a **correctness** (right rows, right labels) and **completeness** (the dataset is present at all) blocker.

---

## 3. P1 — Reliability (Kaggle sessions die unpredictably)

### P1.1 Partial-result checkpointing during inference — tag: **R**

**Problem.** `evaluate_one_model` only writes the final `.npz` after all utterances are processed. A timeout at hour 11 of a 12-h session loses every score computed so far for the current (model, dataset) cell.

**Fix.** Save `<safe_model>_<dataset>.partial.npz` containing `{utt_ids, scores, labels}` every `RUN_CONFIG["partial_save_every"] = 5000` utterances. On resume:
1. Load partial → set of completed `utt_id`s.
2. Filter `eval_df` to only un-finished rows.
3. Run inference on the remainder, append to partial each 5K rows.
4. When all rows done, rename `.partial.npz` → final `.npz` (keys: `utt_ids, scores, labels, eer`).

**Why this matters for the goal.** ASV21 DF on AASIST3 alone is ~3 h of inference. Without partial saves, a single Kaggle hiccup at the 2 h mark restarts that 3 h block from scratch — and the user pays it again on the next session. With partial saves, any session can stop, restart, and resume from within 5K utts of where it left off. This is the **resumability** guarantee.

### P1.2 Persist `utt_ids` in every score file going forward — tag: **C**

**Problem.** The legacy ASV19 pickle and current ASV21 npz files have **no `utt_ids`** → `align_rows_with_result` falls back to order-based alignment, which can silently mis-align grouped EER if protocol parsing order differs by even one row.

**Fix.**
1. Confirm `save_model_result` always populates and saves `utt_ids` (already supported).
2. Add `verify_legacy_alignment(dataset_key, model_name)`: when loading a legacy result without `utt_ids`, spot-check 100 random rows by re-running inference on those few audios and asserting `|fresh - cached| < 1e-3`.
3. If verification fails on any legacy entry → invalidate that cache, re-run.

**Why this matters for the goal.** Order-based alignment is the kind of bug that produces a plausible-looking EER table but with grouped EER columns silently shifted by one row — i.e. you would publish the wrong "AASIST is worst on attack A09" conclusion. This is a **correctness** safety net that costs ~30 s per (model, dataset) cell to run.

### P1.3 Smoke-test mode — tag: **T (debug efficiency)**

**Problem.** A mistake found at minute 5 of a real run can take 7 h to surface. We need a way to validate the full pipeline end-to-end in a few minutes.

**Fix.** Add `RUN_CONFIG["smoke_test_n"]: int | None = None`. When set (e.g. 50), each `evaluate_one_model` only processes the first N rows; `run_error_analysis` is skipped or runs on the truncated frame.

**Why this matters for the goal.** This is a force multiplier for every other change in this list. Without smoke mode, every iteration of "fix something → see if it works" costs hours. With smoke mode, it costs ~5 min, so 7 h of real run is only paid once at the end. Saves time-budget indirectly by preventing wasted long runs caused by trivial bugs.

### P1.4 Pre-flight cell — tag: **C, T (debug efficiency)**

**Problem.** Bugs that only surface during inference (wrong score orientation, missing checkpoint, broken dataset path) waste the budget when caught late.

**Fix.** Insert a cell between Part 4 and Part 5 that:
1. Resolves all dataset paths (calls each locator) and prints a green/red status table.
2. Loads + immediately releases each enabled model — confirms checkpoints exist and load.
3. Runs each model on 5 utterances of each dataset and prints scores — sanity-checks the `predict_fn` orientation (e.g. would have caught the AASIST3 index bug).

**Why this matters for the goal.** P0.1 was a 1-line bug that produced a 79% EER and survived an entire pretrained-evaluation pass. The pre-flight cell is the safety net that catches any future P0.1-class regression in 30 seconds. Protects **correctness** going forward.

### P1.5 Per-model `force_eval` granularity — tag: **R, T**

**Problem.** `RUN_CONFIG["force_eval"][dataset_key]: bool` is per-dataset, not per-model. Re-running only AASIST3 on ASV19 (the typical case after fixing P0.1) requires either re-running all four models on ASV19 or hand-editing the cache.

**Fix.** Replace with nested `RUN_CONFIG["force_eval"][dataset_key][model_name]: bool` and update `evaluate_one_model` to read it as `RUN_CONFIG["force_eval"][dataset_key].get(model_name, False)`.

**Why this matters for the goal.** Surgical re-runs preserve already-validated cache (AASIST + AASIST-L + LFCC+LCNN on ASV19) and only spend time on the entry that actually changed. Saves ~20 min on this specific fix and any future targeted re-run.

---

## 4. P2 — Comparison & visualization (the actual deliverable)

### P2.1 Cross-dataset summary upgrades — tag: **deliverable**

**Problem.** Part 8 / cell 22 currently produces only a CSV table. The goal is a presentation-ready comparison; a flat CSV is hard to communicate.

**Fix.** Add to the existing Part 8:
1. **Heatmap** of EER (rows = models, cols = datasets) — visually shows the codec-robustness story.
2. **Grouped bar chart**: x = dataset, hue = model. One figure for the talk.
3. **Delta columns** `Δ(21−19)` and `Δ(5−19)` per model (formalise the partial logic that's already there).
4. Save to both `cross_dataset_summary.csv` and `cross_dataset_summary.md` (Markdown table) for direct paste into slides / docs.

**Why this matters for the goal.** The whole point of running 4 models × 3 datasets is to *see* the pattern: where does each architecture break down. A CSV alone does not deliver that — the heatmap and grouped bar are the artifacts that go into the final presentation. This is the **deliverable**, not a means to it.

### P2.2 Unified weakness matrix — tag: **deliverable**

**Problem.** Per-dataset grouped EER (by attack, codec, vocoder) is saved separately for each of the 3 datasets. There is no aggregated view of "for each model, what is the single hardest condition across everything we tested."

**Fix.** For each model, gather the 3 worst-EER groups across datasets:

| Model | Worst attack (any dataset) | Worst codec | Worst vocoder |
|-------|---|---|---|
| AASIST | A17 (asv5) — 28% | low_m4a (asv21) — 35% | … |

Store as `cross_dataset_weakness.csv` and a stacked-bar plot.

**Why this matters for the goal.** This is the table that motivates "Propose a more robust approach" (the next phase of the project per `todo.md`). Without it, the proposed improvement has no specific weakness to target. **Direct input** to the next deliverable.

### P2.3 Score correlation across models — tag: **deliverable**

**Problem.** `compute_failure_overlap` runs per-dataset only. There is no aggregate view of "which model pair disagrees most, across everything."

**Fix.** Aggregate `compute_failure_overlap` results across the 3 datasets into one summary. Output: which model pairs are most complementary (low score correlation, low failure overlap).

**Why this matters for the goal.** Justifies the ensemble approach mentioned in `todo.md` — "models that fail on different inputs combine well." Same role as P2.2: feeds the next phase.

---

## 5. P3 — Polish (do last, none are blockers)

### P3.1 Kaggle workflow doc cell — tag: **R (operational)**
Markdown cell at the top of the notebook documenting the upload-as-dataset workflow: run → upload `/kaggle/working/sdd_refactor_results/` as a versioned dataset → re-attach for next session under the slug declared in `ARTIFACT_DATASET_NAMES`. **Why:** without this doc, a future session may forget to attach the result dataset and get the same R-failure as P0.2.

### P3.2 Verbose `find_existing_file` logging — tag: **debug**
Print the resolved path on first hit per group. **Why:** when something does not load, this single log line tells you whether the slug is wrong, the relative path is wrong, or the file simply does not exist yet.

### P3.3 `tqdm` rate limiting — tag: **operational**
Pass `mininterval=2` so Kaggle output buffer does not bloat on long runs. **Why:** Kaggle truncates very long outputs; losing the final EER print after 7 h is annoying but not catastrophic.

### P3.4 Extract helpers to `scripts/sdd_refactor.py` — tag: **maintainability**
Move Part 0–4 helpers out of the notebook once the v2 pipeline is validated end-to-end. **Why:** easier to diff, test, and reuse from a future v3 notebook. Do *after* v2 is green so the notebook stays self-contained during the validation phase.

---

## 6. Time budget for a 12-hour Kaggle session

Estimates assume Kaggle T4, batched inference (P0.3), one model loaded at a time (D3, P0.4):

| Stage | ASV19 (71K) | ASV21 DF (459K) | ASV5 dev (~140K) | Total |
|-------|------------:|----------------:|-----------------:|------:|
| Setup, indexing, LFCC train | — | — | — | ~30 min |
| AASIST eval | 6 min | 38 min | 12 min | ~56 min |
| AASIST-L eval | 6 min | 38 min | 12 min | ~56 min |
| LFCC+LCNN eval | 2 min | 15 min | 5 min | ~22 min |
| **AASIST3 eval** | **30 min** | **~3 h** | **~50 min** | **~4.5 h** |
| Error analysis (per dataset) | 3 min | 5 min | 3 min | ~11 min |
| Cross-dataset summary | — | — | — | ~5 min |
| **Sum** | | | | **~7 h** |

Buffer: ~5 h for slow first-load, model downloads, partial-resume overhead. Comfortable.

If the user later wants ASVspoof 5 **eval** (680K) instead of dev, AASIST3 alone becomes ~5 h on that one dataset → total ~12 h, no buffer. Recommendation: stay on dev for the first complete run.

---

## 7. Suggested implementation order

Each step's choice is justified below.

1. **P0.1** (1-line fix) — cheapest fix with the highest correctness impact; do first so all subsequent runs measure something real. Smoke test on 50 utts → confirm EER drops below 25%.
2. **P0.2** (paths) — needed before *any* cache hit can succeed on Kaggle. Without this, every session is a cold start, and the 12-h budget is not realistic.
3. **P1.3** + **P1.4** (smoke + pre-flight) — short to implement; once these exist, every later change can be validated in 5 min instead of 5 h. Force-multiplier on the rest of the work.
4. **P0.3** (batched inference) — biggest single perf win. Validate against legacy single-sample numbers (EER must match within 0.01% on a 1K-utt subset).
5. **P0.4** (release between models) — unblocks higher AASIST3 batch sizes and prevents OOM once the iteration order changes per D3.
6. **P0.5** (ASV5 indexer) — only after the rest of the pipeline is proven on the two datasets we already have results for. Lowers the risk that an ASV5-specific bug masquerades as a pipeline bug.
7. **P1.1** (partial checkpointing) — turn it on before the first long real run, not before, so checkpoint logic is exercised during smoke phase.
8. **P1.2** (utt_id alignment verify) — run once over legacy caches; act on results.
9. **P1.5** (per-model `force_eval`) — small ergonomics fix; do whenever convenient before the first full run.
10. **P2.x** (visualization) — after results data exists.
11. **P3.x** (polish) — last.

---

## 8. Acceptance check at the end of the first full run

After one 12-h session, the following must all be true:

- `/kaggle/working/sdd_refactor_results/asv19/scores_*.npz` exists for all 4 models, each with `utt_ids`. **(Completeness)**
- Same for `asv21_df/` and `asv5/`. **(Completeness)**
- `cross_dataset_summary.csv` shows AASIST3 EER on ASV19 < 25% (sanity), AASIST EER on ASV19 in [4%, 6%] (matches cached legacy), and a clear EER increase from ASV19 → ASV21 → ASV5 for all four models. **(Correctness)**
- Each dataset folder has `error_analysis/eer_by_attack.{csv,png}` and (where applicable) `eer_by_codec.{csv,png}` and `eer_by_vocoder.{csv,png}`. **(Deliverable)**
- `analysis_summary.json` exists per dataset.
- The whole `sdd_refactor_results/` folder is uploaded as a Kaggle dataset; subsequent sessions resume from cache only. **(Resumability)**

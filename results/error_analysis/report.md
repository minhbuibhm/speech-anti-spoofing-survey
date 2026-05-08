# Error Analysis Report

This report is generated from the current `results.pkl` files and available protocol metadata.
It is the detailed error-analysis companion to `results/report.md`, with grouped breakdowns and interpretation notes.
Labels use `1=bonafide`, `0=spoof`; model scores are bonafide probabilities.

t-DCF is not reported because this repository currently stores CM scores only, not ASV-side scores.
To add t-DCF later, attach the official ASVspoof ASV scores and use the challenge evaluation script.

## Dataset Status

- `asvspoof19`: 71,237 rows, 6 models, utt_ids already present; utt_ids present; metadata joined by utt_id (71,237/71,237 rows matched)
- `asvspoof21`: 458,868 rows, 5 models, utt_id backfill failed: asvspoof21: could not reconstruct evaluated utt_ids. Attempts: metadata_order: length 611,829 != result length 458,868; metadata_head: label mismatch in 31,222/458,868 rows; metadata_filtered_by_existing_flac: no ASVspoof 2021 FLAC folders attached; utt_ids missing; using synthetic row ids; metadata length 611,829 does not align with result length 458,868
- `asvspoof5`: 680,774 rows, 6 models, utt_ids already present; utt_ids present; metadata joined by utt_id (680,774/680,774 rows matched)
- `in_the_wild`: 31,779 rows, 6 models, utt_ids already present; utt_ids present; metadata joined by utt_id (31,779/31,779 rows matched)

## EER Summary

| model         |   asvspoof19 |   asvspoof21 |   asvspoof5 |   in_the_wild |
|:--------------|-------------:|-------------:|------------:|--------------:|
| AASIST        |       4.595  |      17.7005 |     35.7499 |       41.789  |
| AASIST-L      |       6.7445 |      19.1315 |     37.2937 |       45.2799 |
| AASIST3       |      20.8292 |      29.1758 |     38.7481 |       40.1172 |
| LFCC+LCNN     |      19.6421 |      33.81   |     42.3146 |       70.2317 |
| XLS-R+AASIST  |       1.1717 |     nan      |     19.5972 |       10.9071 |
| XLS-R+Nes2Net |       0.449  |       2.9278 |     21.5822 |        5.5737 |

## Generalization Gap vs ASVspoof 2019

| model         |   asvspoof19 |   asvspoof21 |   asvspoof5 |   in_the_wild |
|:--------------|-------------:|-------------:|------------:|--------------:|
| AASIST        |            0 |      13.1055 |     31.155  |       37.1941 |
| AASIST-L      |            0 |      12.387  |     30.5493 |       38.5354 |
| AASIST3       |            0 |       8.3466 |     17.9189 |       19.288  |
| LFCC+LCNN     |            0 |      14.1678 |     22.6725 |       50.5895 |
| XLS-R+AASIST  |            0 |     nan      |     18.4256 |        9.7354 |
| XLS-R+Nes2Net |            0 |       2.4788 |     21.1333 |        5.1248 |

## Robustness Summary

| model         |   mean_eer |   std_eer |   datasets_evaluated |
|:--------------|-----------:|----------:|---------------------:|
| XLS-R+Nes2Net |     7.6332 |    9.5319 |                    4 |
| XLS-R+AASIST  |    10.5587 |    9.2177 |                    3 |
| AASIST        |    24.9586 |   17.0007 |                    4 |
| AASIST-L      |    27.1124 |   17.4381 |                    4 |
| AASIST3       |    32.2176 |    9.0185 |                    4 |
| LFCC+LCNN     |    41.4996 |   21.3157 |                    4 |

## Notes

- ASVspoof 5 now uses Track 1 eval (680,774 rows), not the older dev subset.
- ASVspoof 5 is the strongest modern-attack stress test in the current survey: all models show a large gap relative to ASVspoof 2019 LA.
- XLS-R based systems remain lower on ASVspoof 5 and In-the-Wild than non-SSL systems, but ASVspoof 5 EER is still around 20%, so SSL front-ends do not fully solve modern attack/codec robustness.
- `XLS-R+AASIST` is available for ASVspoof 5 and In-the-Wild; ASVspoof 2021 DF remains pending.
- `hard_errors.csv` includes universal hard examples, but some universal rows do not carry full attack/codec metadata, so use them for qualitative inspection rather than detailed grouped conclusions.

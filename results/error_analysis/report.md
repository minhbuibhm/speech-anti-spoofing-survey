# Error Analysis Report

This report is generated from the current `results.pkl` files and available protocol metadata.
Labels use `1=bonafide`, `0=spoof`; model scores are bonafide probabilities.

t-DCF is not reported because this repository currently stores CM scores only, not ASV-side scores.
To add t-DCF later, attach the official ASVspoof ASV scores and use the challenge evaluation script.

## Dataset Status

- `asvspoof19`: 71,237 rows, 6 models, metadata aligned
- `asvspoof21`: 458,868 rows, 5 models, metadata length 611,829 does not align with result length 458,868
- `asvspoof5`: 140,950 rows, 6 models, metadata length 680,774 does not align with result length 140,950
- `in_the_wild`: 31,779 rows, 5 models, metadata aligned

## EER Summary

| model         |   asvspoof19 |   asvspoof21 |   asvspoof5 |   in_the_wild |
|:--------------|-------------:|-------------:|------------:|--------------:|
| AASIST        |       4.595  |      17.7005 |     37.812  |       41.789  |
| AASIST-L      |       6.7445 |      19.1315 |     39.4685 |       45.2799 |
| AASIST3       |      20.8292 |      29.1758 |     19.0307 |       40.1172 |
| LFCC+LCNN     |      19.6421 |      33.81   |     22.5996 |       70.2317 |
| XLS-R+AASIST  |       1.1717 |     nan      |      2.5519 |      nan      |
| XLS-R+Nes2Net |       0.449  |       2.9278 |      1.8034 |        5.5737 |

## Generalization Gap vs ASVspoof 2019

| model         |   asvspoof19 |   asvspoof21 |   asvspoof5 |   in_the_wild |
|:--------------|-------------:|-------------:|------------:|--------------:|
| AASIST        |            0 |      13.1055 |     33.217  |       37.1941 |
| AASIST-L      |            0 |      12.387  |     32.724  |       38.5354 |
| AASIST3       |            0 |       8.3466 |     -1.7985 |       19.288  |
| LFCC+LCNN     |            0 |      14.1678 |      2.9575 |       50.5895 |
| XLS-R+AASIST  |            0 |     nan      |      1.3803 |      nan      |
| XLS-R+Nes2Net |            0 |       2.4788 |      1.3544 |        5.1248 |

## Robustness Summary

| model         |   mean_eer |   std_eer |   datasets_evaluated |
|:--------------|-----------:|----------:|---------------------:|
| XLS-R+AASIST  |     1.8618 |    0.976  |                    2 |
| XLS-R+Nes2Net |     2.6885 |    2.1742 |                    4 |
| AASIST        |    25.4741 |   17.462  |                    4 |
| AASIST3       |    27.2882 |    9.6272 |                    4 |
| AASIST-L      |    27.6561 |   17.8894 |                    4 |
| LFCC+LCNN     |    36.5708 |   23.2555 |                    4 |

## Notes

- Add human interpretation here after reviewing grouped errors and hard examples.

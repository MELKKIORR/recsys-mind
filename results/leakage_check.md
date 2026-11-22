# Leakage check

Artefacts: 2026-11-22_0004b35

| Check | Result | Detail |
| --- | --- | --- |
| Temporal split | PASS | train 2019-11-09 00:01 to 2019-11-13 23:52, validation from 2019-11-14 00:04, test from 2019-11-15 00:05 |
| No shared impression | PASS | 0 impressions in more than one split |
| Graph edges from train only | PASS | saved entity-entity matrix vs rebuilt from train: 40 vs 40 entries, total difference 0 |
| Popularity from train only | PASS | 300 articles with train clicks |
| Ignored lists from train only | PASS | 40 readers with ignored articles |
| No val/test click in popularity | PASS | 139 articles clicked only after training, 0 with a popularity > 0 |
| Test split used once | PASS | not used yet |

Overall: PASS
Hyperparameters are tuned on the validation split only; each test impression uses only its own history (the MIND history field, clicks before the impression).
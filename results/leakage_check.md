# Leakage check

Artefacts: 2026-11-15_nogit

| Check | Result | Detail |
| --- | --- | --- |
| Temporal split | PASS | train 2019-11-09 00:00 to 2019-11-13 23:59, validation from 2019-11-14 00:00, test from 2019-11-15 00:00 |
| No shared impression | PASS | 0 impressions in more than one split |
| Graph edges from train only | PASS | saved entity-entity matrix vs rebuilt from train: 204492 vs 204492 entries, total difference 0 |
| Popularity from train only | PASS | 6398 articles with train clicks |
| Ignored lists from train only | PASS | 46012 readers with ignored articles |
| No val/test click in popularity | PASS | 2702 articles clicked only after training, 0 with a popularity > 0 |
| Test split used once | PASS | not used yet |

Overall: PASS
Hyperparameters are tuned on the validation split only; each test impression uses only its own history (the MIND history field, clicks before the impression).
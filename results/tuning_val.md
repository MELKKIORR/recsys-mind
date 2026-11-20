# Week 3 tuning on validation (MLE-Ops)

Validation split, holdout readers only, 1000 impressions per point (seed 1), metric NDCG@10. LLM: mock .

| Parameter | Values tried (NDCG@10) | Best |
| --- | --- | --- |
| hops | 1: 0.3112, 2: 0.3111, 3: 0.3125 | 3 |
| damping | 0.5: 0.3140, 0.65: 0.3132, 0.75: 0.3141, 0.85: 0.3125 | 0.75 |
| recency_decay | 0.7: 0.3007, 0.8: 0.3086, 0.9: 0.3141, 1.0: 0.3231 | 1.0 |
| ignored_weight | 0.0: 0.3263, 0.25: 0.3307, 0.5: 0.3231, 1.0: 0.2891 | 0.25 |
| category_weight | 0.0: 0.2588, 0.15: 0.3235, 0.3: 0.3307, 0.5: 0.3264 | 0.3 |
| fusion_weight (rule) | 0.0: 0.3307, 0.25: 0.3307, 0.5: 0.3307, 0.75: 0.3307, 1.0: 0.3307 | 0.0 |
| L2 of the learned layer | 0.0: 0.3278, 0.001: 0.3282, 0.01: 0.3286 | 0.01 |

## Final: 3 draws of 2000 impressions (seeds [1, 2, 3])

| Model | NDCG@10 | recall@20 | AUC | MRR | NDCG@5 |
| --- | --- | --- | --- | --- | --- |
| random | 0.2601 ± 0.0026 | 0.6846 ± 0.0058 | 0.4987 ± 0.0024 | 0.2004 ± 0.0011 | 0.2000 ± 0.0016 |
| popular | 0.2759 ± 0.0047 | 0.6862 ± 0.0089 | 0.5129 ± 0.0019 | 0.2127 ± 0.0043 | 0.2144 ± 0.0039 |
| graph (default) | 0.3263 ± 0.0054 | 0.7436 ± 0.0060 | 0.5685 ± 0.0056 | 0.2561 ± 0.0056 | 0.2675 ± 0.0076 |
| graph (tuned) | 0.3417 ± 0.0096 | 0.7584 ± 0.0084 | 0.5977 ± 0.0092 | 0.2671 ± 0.0085 | 0.2837 ± 0.0098 |
| full, fixed rule | 0.3417 ± 0.0096 | 0.7584 ± 0.0084 | 0.5970 ± 0.0093 | 0.2671 ± 0.0085 | 0.2837 ± 0.0098 |
| full, learned fusion | 0.3408 ± 0.0095 | 0.7584 ± 0.0084 | 0.5958 ± 0.0092 | 0.2660 ± 0.0086 | 0.2809 ± 0.0099 |

Figures: figures/tuning_val.png (one parameter at a time), figures/fusion_loss.png (train and validation loss of the fusion layer). Best values written to configs/tuned.yaml.

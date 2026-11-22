# LLM call time and cost (mock , top 10)

| Call | Median time (s) | Max (s) | Input tokens | Output tokens | Cost per call ($) |
| --- | --- | --- | --- | --- | --- |
| rerank no cache | 0.0 | 0.0 | 510.2 | 57.2 | 0.00000 |
| rerank cached | 0.02 | 0.022 | 0.0 | 0.0 | 0.00000 |
| summary no cache | 0.0 | 0.0 | 400.4 | 14.0 | 0.00000 |
| summary cached | 0.021 | 0.025 | 0.0 | 0.0 | 0.00000 |

Re-ranking 1,000 impressions costs about $0.0, the evaluation (3 seeds x 2,000) about $0.0 and 0.0 minutes without cache. A cached call costs nothing, so reruns and the website use the cache.

# LLM re-ranking (Week 3, MLE-Core)

Choose the LLM in `configs/default.yaml` (section `llm`) and put the key in an environment variable,
never in the code:

| Provider | Config | Key (PowerShell) |
| --- | --- | --- |
| No LLM, for tests | `provider: mock` | none |
| Claude | `provider: anthropic`, `model: <model name>` | `$env:ANTHROPIC_API_KEY="..."` |
| OpenAI, Mistral, Groq, Ollama | `provider: openai`, `model: ...`, `base_url: ...` | `$env:LLM_API_KEY="..."` |

Answers are cached in `results/llm_cache/`, so a rerun costs nothing. Ablations and innovation metrics:

```bash
python -m src.eval.evaluate --split val --models popular graph graph_no_ignored graph_1hop llm_only full --sample 2000 --seeds 1 2 3
python scripts/sample_users.py           # 3 sample readers for the qualitative check
```

The results file also holds the LLM fallback rate (invalid answers), the number of unknown ids and the
explanation fidelity (share of explained articles that drop when their evidence entity is removed).


## MLE-Ops (Week 3): cost, profile summaries, fusion layer, tuning

```bash
python scripts/llm_benchmark.py --calls 5      # time and cost of one call, results/llm_benchmark.md
python -m src.llm.profile_summary --users 20   # LLM profile summaries, results/profile_summaries.json
python -m src.llm.fusion --train 3000 --val 1000   # fusion layer, results/fusion_weights.json + loss curves
python scripts/tune.py --sample 1000 --final-sample 2000 --seeds 1 2 3   # search on val + 3 draws
```

Prices for the cost measure go in `llm.price_input_per_mtok` and `llm.price_output_per_mtok`.
The fusion layer and the search only use the validation split: readers are split in two fixed halves,
"fit" readers train the layer and "holdout" readers give the validation loss and every tuning number.
Set `rerank.fusion: learned` to use the trained layer in the API and in `src.eval.evaluate`.

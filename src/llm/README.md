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


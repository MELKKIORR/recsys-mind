# News companion: explained news recommendations on MIND

ING5 Groupe 01, ECE Paris. Melchior Gerrier (PL), Ines Yaici (DE), Jade Vigouroux (MLE-Core),
Adam Kheloufi (MLE-Ops), Eden Masmoudi (SE).

We recommend news articles with an entity graph built from MIND (Wikidata entities), a multi-hop
personalised PageRank that keeps an evidence path for each article ("Because you read X: Tesla -> Panasonic"),
and, from Week 3, an LLM that re-ranks the top 10 under code checks.

## Run it locally (about 5 min, no API key needed)

Requirements: Python 3.11 or 3.12, and Git. Tested on Windows 11 (PowerShell) and Linux.

### 1. Install (terminal 1)

```bash
git clone https://github.com/MELKKIORR/recsys-mind.git
cd recsys-mind
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\activate` (if scripts are blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first)
- macOS / Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
```

### 2. Prepare the data (about 1 min)

```bash
python scripts/download_mind.py          # MIND small into data/raw/ (research use only)
python -m src.data.preprocess            # cleaning + split, prints the data stats
python -m src.data.db                    # SQLite database data/recsys.db
python -m src.graph.build_graph          # graph + artefacts in results/artefacts/
```

### 3. Start the API (terminal 1, keep it open)

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

http://127.0.0.1:8000/health should show `"status": "ok"`. API docs: http://127.0.0.1:8000/docs.

### 4. Start the website (terminal 2)

Open a second terminal, go to the `recsys-mind` folder and activate the virtual environment again
(same command as in step 1), then:

```bash
streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501, pick a demo reader, then like an article: the next top 10 changes.
On Windows use `127.0.0.1`, not `localhost` (`localhost` tries IPv6 first and adds about 2 s per call).

### No API key needed

Without a key, recommendations come from the graph model with their "Because you read X" explanation;
only the LLM re-ranking of the top 10 is skipped. To enable it, set the key before step 3
(`$env:LLM_API_KEY="..."` in PowerShell, `export LLM_API_KEY=...` on Linux/macOS) and see `src/llm/README.md`.
The LLM answer is read from the disk cache (`results/llm_cache/`), never called while the reader waits:
on a cache miss the graph ranking is served and the LLM is called in the background, so the next request
of that reader is re-ranked (`LLM_CACHE_ONLY=0` calls it live instead). `PREDICT_MODE=mock` gives the
popular-articles mock of Week 2.

### If the MIND download fails

Run everything on a small fake dataset in the MIND format (a few seconds). Replace steps 2 and 3 with:

```bash
python scripts/make_toy_data.py
python -m src.data.preprocess configs/test.yaml
python -m src.data.db configs/test.yaml
python -m src.graph.build_graph configs/test.yaml
```

Then start the API with the test config:

- Windows PowerShell: `$env:RECSYS_CONFIG="configs/test.yaml"; uvicorn api.main:app --host 127.0.0.1 --port 8000`
- macOS / Linux: `RECSYS_CONFIG=configs/test.yaml uvicorn api.main:app --host 127.0.0.1 --port 8000`

Step 4 is unchanged.

### With Docker

```bash
docker compose --profile setup run --rm prepare     # once: download MIND, preprocess, database, graph
docker compose up --build                            # API on :8000, website on http://127.0.0.1:8501
```

`data/` and `results/` are mounted from the host, so the data prepared without Docker is reused as is.
Pass the LLM key with `ANTHROPIC_API_KEY=... docker compose up` (or `LLM_API_KEY`).
Set `GIT_COMMIT` (`$env:GIT_COMMIT = git rev-parse --short HEAD` in PowerShell) before `prepare` so the graph
version carries the commit; without it the version ends in `_nogit`, since `.git` is not copied into the image.
Tested on Docker Desktop 29.9 (Windows 11) from a fresh clone: `prepare` takes about 2 min, then both services
are healthy within 15 s.

### Response time

```bash
python scripts/latency_check.py --users 20 --rounds 2      # the API must be running
```

Times 20 demo readers twice (cold then warm LLM cache), checks that a like sent to `/log_interaction` is in
the `Interactions` table and changes the next top 10, and writes `results/latency_<date>.md`. Every API
response also carries its server time in the `X-Response-Time-ms` header; `GET /latency` summarises them.

## Evaluate

```bash
python -m src.eval.evaluate --split val --models random popular graph graph_no_ignored graph_1hop --sample 2000 --seeds 1 2 3
```

Metrics per impression (AUC, MRR, NDCG@5, NDCG@10, Recall@20), averaged, with separate results for
impressions that contain a cold-start article and for readers with fewer than 5 clicks.
Results are saved in `results/`.

## Final evaluation, leakage check and report tables (PL)

All tuning is done on the validation split. The test split is used once, for the final systems:

```bash
python -m src.eval.evaluate --split test --final --models random popular graph graph_no_ignored graph_1hop full --sample 2000 --seeds 1 2 3
python -m src.eval.leakage_check         # results/leakage_check.md, every line must be PASS
python -m src.eval.report_tables         # results/report_tables.md and results/figures/ablations_*.png
```

Without `--final` the test split is refused; a second run needs `--force` and is logged in
`results/final_test_log.json`. Cold-start NDCG@10 ranks only the articles never seen in training,
among themselves, inside each impression.

## Tests

```bash
python -m pytest -q
```

## Layout

| Folder | Content | Owner |
| --- | --- | --- |
| `scripts/` | MIND download, toy data | DE |
| `src/data/` | preprocessing and split, SQLite | DE, PL |
| `src/graph/` | graph construction, multi-hop retrieval, evidence paths | PL, MLE-Core |
| `src/eval/` | metrics and evaluation harness | PL |
| `src/llm/` | LLM profile summary and re-ranking (Week 3) | MLE-Core, MLE-Ops |
| `src/predict.py` | predict(user_id, k) used by the API | MLE-Ops |
| `api/`, `app/` | FastAPI backend, Streamlit frontend | SE |

## No leakage

MIND train period minus its last day is the training set, its last day is the validation set, MIND dev is
the test set. Entity-entity edges, popularity and ignored articles come from the training set only; a test
impression only uses its own history.

# News companion: explained news recommendations on MIND

ING5 Groupe 01, ECE Paris. Melchior Gerrier (PL), Ines Yaici (DE), Jade Vigouroux (MLE-Core),
Adam Kheloufi (MLE-Ops), Eden Masmoudi (SE).

We recommend news articles with an entity graph built from MIND (Wikidata entities), a multi-hop
personalised PageRank that keeps an evidence path for each article ("Because you read X: Tesla -> Panasonic"),
and, from Week 3, an LLM that re-ranks the top 10 under code checks.

## Run it locally

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_mind.py          # MIND small into data/raw/ (research use only)
python -m src.data.preprocess            # cleaning + split, prints the data stats
python -m src.data.db                    # SQLite database data/recsys.db
python -m src.graph.build_graph          # graph + artefacts in results/artefacts/

uvicorn api.main:app --host 0.0.0.0 --port 8000        # API, docs at http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py                     # website at http://127.0.0.1:8501, second terminal
```

The API serves the graph model with the LLM re-ranking of the top 10 (`PREDICT_MODE=mock` gives the
popular-articles mock of Week 2). The LLM answer is read from the disk cache (`results/llm_cache/`), never
called while the reader waits: on a cache miss the graph ranking is served and the LLM is called in the
background, so the next request of that reader is re-ranked (`LLM_CACHE_ONLY=0` calls it live instead).
Choose the LLM in `configs/default.yaml` (see `src/llm/README.md`).

On Windows use `127.0.0.1`, not `localhost`: `localhost` tries IPv6 first and adds about 2 s to every call.
Without the real data, `python scripts/make_toy_data.py` writes a small fake dataset in the MIND format in
`data/toy/raw`; add `configs/test.yaml` after each `python -m ...` command to run the pipeline on it, and
start the API with `RECSYS_CONFIG=configs/test.yaml`.

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
python -m src.eval.evaluate --split val --models random popular graph --sample 2000 --seeds 1 2 3
```

Metrics per impression (AUC, MRR, NDCG@5, NDCG@10, Recall@20), averaged, with separate results for
impressions that contain a cold-start article and for readers with fewer than 5 clicks.
Results are saved in `results/`.

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

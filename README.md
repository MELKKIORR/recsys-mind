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

uvicorn api.main:app --host 0.0.0.0 --port 8000        # API, docs at http://localhost:8000/docs
streamlit run app/streamlit_app.py                     # frontend, in a second terminal
```

`PREDICT_MODE=graph uvicorn api.main:app ...` serves the graph model instead of the popular-articles mock.
Without the real data, `python scripts/make_toy_data.py` writes a small fake dataset in the MIND format in
`data/toy/raw`; add `configs/test.yaml` after each `python -m ...` command to run the pipeline on it.

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

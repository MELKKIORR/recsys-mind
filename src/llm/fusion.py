"""Learned fusion layer (MLE-Ops): mixes the graph score and the LLM rank of the top N candidates.

Week 3 replaces the fixed rule "(1-w) * graph position + w * LLM position" by a small logistic layer
trained on clicks. For each of the top N graph candidates of an impression it uses 7 features:

  graph_z         graph score, standardised inside the top N
  graph_pos       graph position, 1 = first, 0 = last
  llm_pos         LLM position, 1 = first, 0 = last (equal to graph_pos when the LLM falls back)
  popularity      log(1 + train clicks), divided by the largest value
  cold            1 if the article was never clicked in training
  category_aff    weight of the article category in the reader profile
  category_ign    weight of the article category in the ignored articles

score = sigmoid(w . x + b), trained with binary cross-entropy (clicked or not) and Adam, by mini-batches
of impressions, with early stopping on the validation loss. Only the order of the top N changes; the
other candidates keep the graph order.

Data and leakage: popularity and ignored lists come from the train impressions, so training the layer
on those impressions would let it learn from its own labels. It is trained on the validation split,
split by reader into two fixed halves: "fit" readers train the layer, "holdout" readers give the
validation loss curve and every validation number of the hyper-parameter search. The test split is
never read here.

    python -m src.llm.fusion [config] --train 3000 --val 1000      # weights + loss curves in results/
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.eval.metrics import ndcg

FEATURES = ["graph_z", "graph_pos", "llm_pos", "popularity", "cold", "category_aff", "category_ign"]


def reader_half(user_id: str) -> str:
    """Fixed split of the validation readers: 'fit' (trains the layer) or 'holdout'."""
    return "fit" if int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def val_half(beh: pd.DataFrame, half: str) -> pd.DataFrame:
    return beh[beh["user_id"].map(reader_half) == half]


def features(rec, profile, top: list, graph_scores, llm_order: list) -> np.ndarray:
    """Feature matrix (len(top), 7). `top` is in graph order, `llm_order` holds the same ids."""
    n = len(top)
    g = np.asarray(graph_scores, dtype=float)
    pos = np.linspace(1, 0, n) if n > 1 else np.ones(n)
    llm_rank = {a: r for r, a in enumerate(llm_order)}
    llm_pos = np.array([1 - llm_rank.get(a, r) / (n - 1) if n > 1 else 1.0 for r, a in enumerate(top)])
    max_pop = getattr(rec, "_log_max_pop", None)
    if max_pop is None:
        max_pop = rec._log_max_pop = np.log1p(max(rec.popularity.values(), default=0)) or 1.0
    pop = np.array([rec.popularity.get(a, 0) for a in top], dtype=float)
    cats = [rec.category.get(a) for a in top]
    return np.column_stack([
        (g - g.mean()) / (g.std() + 1e-9),
        pos,
        llm_pos,
        np.log1p(pop) / max_pop,
        (pop == 0).astype(float),
        [profile.category_w.get(c, 0.0) for c in cats],
        [profile.ignored_category_w.get(c, 0.0) for c in cats],
    ])


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class FusionLayer:
    def __init__(self, w=None, b=0.0, mean=None, std=None):
        k = len(FEATURES)
        self.w = np.zeros(k) if w is None else np.asarray(w, dtype=float)
        self.b = float(b)
        self.mean = np.zeros(k) if mean is None else np.asarray(mean, dtype=float)
        self.std = np.ones(k) if std is None else np.asarray(std, dtype=float)

    def score(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(((X - self.mean) / self.std) @ self.w + self.b)

    def order(self, rec, profile, top, graph_scores, llm_order) -> list:
        s = self.score(features(rec, profile, top, graph_scores, llm_order))
        return [top[i] for i in np.argsort(-s, kind="stable")]

    # training ---------------------------------------------------------------
    def loss(self, groups, l2=0.0) -> float:
        X = np.vstack([g[0] for g in groups])
        y = np.concatenate([g[1] for g in groups])
        p = np.clip(self.score(X), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + l2 * (self.w ** 2).sum())

    def ndcg10(self, groups) -> float:
        """NDCG@10 of the whole impression: fused top N, then the rest in graph order."""
        vals = []
        for X, y_top, y_rest in groups:
            labels = np.concatenate([y_top[np.argsort(-self.score(X), kind="stable")], y_rest])
            v = ndcg(labels, -np.arange(len(labels)), 10)
            if not np.isnan(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def fit(self, train, val, epochs=60, lr=0.01, l2=1e-3, batch=64, patience=8, seed=0, log=print):
        """Adam on the mean BCE of mini-batches of impressions. Keeps the epoch with the best val loss."""
        rng = np.random.default_rng(seed)
        X_all = np.vstack([g[0] for g in train])
        self.mean, self.std = X_all.mean(0), X_all.std(0) + 1e-9
        m, v = np.zeros(len(FEATURES) + 1), np.zeros(len(FEATURES) + 1)
        b1, b2, t = 0.9, 0.999, 0
        history = [{"epoch": 0, "train_loss": self.loss(train, l2), "val_loss": self.loss(val, l2),
                    "val_ndcg@10": self.ndcg10(val)}]
        best, best_state, wait = history[0]["val_loss"], (self.w.copy(), self.b), 0
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(len(train))
            for s in range(0, len(idx), batch):
                Xb = np.vstack([train[i][0] for i in idx[s:s + batch]])
                yb = np.concatenate([train[i][1] for i in idx[s:s + batch]])
                Z = (Xb - self.mean) / self.std
                err = _sigmoid(Z @ self.w + self.b) - yb
                grad = np.append(Z.T @ err / len(yb) + 2 * l2 * self.w, err.mean())
                t += 1
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad ** 2
                step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
                self.w -= step[:-1]
                self.b -= step[-1]
            h = {"epoch": epoch, "train_loss": self.loss(train, l2), "val_loss": self.loss(val, l2),
                 "val_ndcg@10": self.ndcg10(val)}
            history.append(h)
            log(f"epoch {epoch:3d}  train {h['train_loss']:.4f}  val {h['val_loss']:.4f}  "
                f"val NDCG@10 {h['val_ndcg@10']:.4f}")
            if h["val_loss"] < best - 1e-5:
                best, best_state, wait = h["val_loss"], (self.w.copy(), self.b), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        self.w, self.b = best_state
        return history

    # save / load ------------------------------------------------------------
    def to_dict(self):
        return {"features": FEATURES, "w": self.w.round(6).tolist(), "b": round(self.b, 6),
                "mean": self.mean.round(6).tolist(), "std": self.std.round(6).tolist()}

    def save(self, path, **extra):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.to_dict(), **extra}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        if d.get("features") != FEATURES:
            raise ValueError(f"{path} was trained with other features: {d.get('features')}")
        return cls(d["w"], d["b"], d["mean"], d["std"])


def build_groups(rec, reranker, behaviors: pd.DataFrame, top_n=10) -> list:
    """One (X, labels of the top N, labels of the rest in graph order) per impression.
    `reranker` must have fusion_weight 1 and no fusion layer, so its order is the LLM order."""
    groups = []
    for row in behaviors.itertuples(index=False):
        profile = rec.build_profile(row.history, rec.ignored.get(row.user_id, []))
        cands = list(row.candidates)
        g = rec.score(profile, cands)
        order = list(np.argsort(-g, kind="stable"))
        top_i, rest_i = order[:top_n], order[top_n:]
        top = [cands[i] for i in top_i]
        llm_order = reranker.rerank(profile, top, g[top_i])["order"]
        labels = np.asarray(row.labels, dtype=float)
        groups.append((features(rec, profile, top, g[top_i], llm_order), labels[top_i], labels[rest_i]))
    return groups


def plot_curves(history, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    a1.plot(ep, [h["train_loss"] for h in history], label="train (fit readers)")
    a1.plot(ep, [h["val_loss"] for h in history], label="validation (holdout readers)")
    a1.set_xlabel("epoch"), a1.set_ylabel("binary cross-entropy"), a1.legend(frameon=False)
    a2.plot(ep, [h["val_ndcg@10"] for h in history], color="tab:green")
    a2.set_xlabel("epoch"), a2.set_ylabel("validation NDCG@10")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title or "Fusion layer training")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_reranker_llm_order(cfg, rec, client):
    from src.llm.rerank import Reranker
    r = cfg["rerank"]
    return Reranker(rec, client, r["top_n"], 1.0, r["min_valid_share"], cfg["llm"]["prompt_version"])


def train_from_config(cfg, n_train=3000, n_val=1000, seed=1, rec=None, client=None, log=print, **fit_args):
    from src.graph.retrieval import GraphRecommender
    from src.llm.client import LLMClient
    rec = rec or GraphRecommender.from_config(cfg)
    client = client or LLMClient.from_config(cfg)
    rr = make_reranker_llm_order(cfg, rec, client)
    beh = pd.read_pickle(cfg["processed_dir"] / "behaviors_val.pkl")
    beh = beh[beh["labels"].map(sum) > 0]
    fit_beh, hold_beh = val_half(beh, "fit"), val_half(beh, "holdout")
    fit_beh = fit_beh.sample(n=min(n_train, len(fit_beh)), random_state=seed)
    hold_beh = hold_beh.sample(n=min(n_val, len(hold_beh)), random_state=seed)
    t0 = time.time()
    top_n = cfg["rerank"]["top_n"]
    train, val = build_groups(rec, rr, fit_beh, top_n), build_groups(rec, rr, hold_beh, top_n)
    log(f"features: {len(train)} fit impressions, {len(val)} holdout impressions "
        f"({time.time() - t0:.0f} s, LLM {client.stats})")
    layer = FusionLayer()
    history = layer.fit(train, val, log=log, **fit_args)
    graph_only = FusionLayer(w=np.eye(len(FEATURES))[1] * 10)          # sorts by graph position
    graph_only.mean, graph_only.std = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
    return layer, history, {"n_fit": len(train), "n_holdout": len(val), "llm_provider": client.provider,
                            "llm_model": client.model, "llm_fallback_rate": round(rr.fallback_rate(), 4),
                            "holdout_ndcg@10_graph_order": round(graph_only.ndcg10(val), 4),
                            "holdout_ndcg@10_fusion": round(layer.ndcg10(val), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/default.yaml")
    ap.add_argument("--train", type=int, default=3000, help="impressions of the 'fit' readers")
    ap.add_argument("--val", type=int, default=1000, help="impressions of the 'holdout' readers")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    layer, history, info = train_from_config(cfg, args.train, args.val, args.seed, epochs=args.epochs,
                                             lr=args.lr, l2=args.l2)
    out = ROOT / cfg["rerank"].get("fusion_weights", "results/fusion_weights.json")
    layer.save(out, training={**info, "lr": args.lr, "l2": args.l2, "epochs_run": len(history) - 1},
               history=history)
    fig = out.parent / "figures" / "fusion_loss.png"
    plot_curves(history, fig)
    print("weights:", dict(zip(FEATURES, layer.w.round(3))), "b", round(layer.b, 3))
    print(info)
    print("Saved", out, "and", fig)


if __name__ == "__main__":
    main()

"""Which entities are kept out of the explanations? (MLE-Core)

    python scripts/hub_entities.py [config] [--show "Eastern Time Zone" "Twitter" ...]

Lists the most frequent entities with their number of articles and Wikidata type, the entities given
with --show, and whether each one is blocked as an evidence path (retrieval.path_stop_entities in the
config, or the default list in src/graph/retrieval.py). Blocked entities still count in the scores.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.graph.retrieval import GraphRecommender  # noqa: E402


def main(argv):
    show = []
    if "--show" in argv:
        i = argv.index("--show")
        show, argv = argv[i + 1:], argv[:i]
    cfg = load_config(argv[0] if argv else "configs/default.yaml")
    rec = GraphRecommender.from_config(cfg)
    types = rec.entities["type"].tolist() if "type" in rec.entities else [""] * len(rec.ent_labels)
    order = list(np.argsort(-rec.entity_df)[:30])
    order += [j for j, l in enumerate(rec.ent_labels) if (l in show or rec.path_blocked[j]) and j not in order]
    print(f"{'entity':40s} {'articles':>8s} {'type':>5s}  path")
    for j in order:
        print(f"{rec.ent_labels[j][:40]:40s} {int(rec.entity_df[j]):8d} {str(types[j]):>5s}  "
              f"{'BLOCKED' if rec.path_blocked[j] else 'kept'}")
    missing = [l for l in show if l not in rec.ent_labels]
    if missing:
        print("Not found:", ", ".join(missing))
    print(f"\n{int(rec.path_blocked.sum())} entities blocked as paths. Scores are not affected.")


if __name__ == "__main__":
    main(sys.argv[1:])

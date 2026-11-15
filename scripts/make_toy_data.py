"""Write a small fake dataset in the MIND format into data/raw/ (for tests and CI only)."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATS = {"autos": ["Tesla", "Panasonic", "Ford", "Toyota"],
        "finance": ["Tesla", "Apple", "Federal Reserve", "Panasonic"],
        "sports": ["NFL", "LeBron James", "Los Angeles Lakers"],
        "politics": ["Donald Trump", "Joe Biden", "Congress"],
        "tech": ["Apple", "Google", "Microsoft"]}


def entity(label):
    return {"Label": label, "Type": "O", "WikidataId": "Q" + str(abs(hash(label)) % 100000),
            "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": [label]}


def write_split(out_dir, start, days, n_news, n_imps, news_offset, rng, users):
    out_dir.mkdir(parents=True, exist_ok=True)
    news = []
    for i in range(n_news):
        cat = rng.choice(list(CATS))
        ents = rng.sample(CATS[cat], k=rng.randint(0, 2))
        nid = f"N{news_offset + i}"
        news.append((nid, cat, ents))
    with open(out_dir / "news.tsv", "w") as f:
        for nid, cat, ents in news:
            title = f"{cat} story about {', '.join(ents) or 'nothing'}"
            f.write("\t".join([nid, cat, cat + "_sub", title, "abstract", "http://x",
                               json.dumps([entity(e) for e in ents]), "[]"]) + "\n")
    with open(out_dir / "behaviors.tsv", "w") as f:
        for i in range(n_imps):
            user = rng.choice(users)
            fav = users.index(user) % len(CATS)
            fav_cat = list(CATS)[fav]
            t = start + timedelta(minutes=rng.randint(0, days * 24 * 60 - 1))
            hist = [n[0] for n in rng.sample(news, 5) if n[1] == fav_cat] or [news[0][0]]
            shown = rng.sample(news, 8)
            labels = [int(n[1] == fav_cat and rng.random() < 0.7) for n in shown]
            if sum(labels) == 0:
                labels[0] = 1
            imp = " ".join(f"{n[0]}-{l}" for n, l in zip(shown, labels))
            f.write("\t".join([str(i + 1), user, t.strftime("%m/%d/%Y %I:%M:%S %p"),
                               " ".join(hist), imp]) + "\n")


def main():
    import sys
    rng = random.Random(0)
    users = [f"U{i}" for i in range(40)]
    raw = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/toy/raw")
    write_split(raw / "train", datetime(2019, 11, 9), 6, 300, 1500, 0, rng, users)
    write_split(raw / "dev", datetime(2019, 11, 15), 1, 150, 300, 1000, rng, users)
    print("Toy MIND data written to", raw)


if __name__ == "__main__":
    main()

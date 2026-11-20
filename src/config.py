from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_dir", "processed_dir", "db_path", "artefacts_dir"):
        cfg[key] = ROOT / cfg[key]
    if "llm" in cfg:
        cfg["llm"]["cache_dir"] = ROOT / cfg["llm"]["cache_dir"]
    return cfg

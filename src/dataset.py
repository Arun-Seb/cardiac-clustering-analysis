"""
dataset.py
──────────
Data loading for the PASCAL Heartbeat Sounds dataset.
"""

import os
import numpy as np
import librosa
from pathlib import Path

SR       = 16000
DURATION = 4
VALID    = {"normal", "murmur", "extrastole", "artifact", "extrahls"}


def find_dataset(base: Path) -> dict:
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4:
            continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths:
                paths["set_a_csv"] = root / name
            if name == "set_b.csv" and "set_b_csv" not in paths:
                paths["set_b_csv"] = root / name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths:
                paths["set_a_dir"] = root / d
            if d == "set_b" and "set_b_dir" not in paths:
                paths["set_b_dir"] = root / d
    missing = [k for k in ("set_a_csv","set_b_csv","set_a_dir","set_b_dir")
               if k not in paths]
    if missing:
        raise FileNotFoundError(f"Missing dataset components: {missing}")
    return paths


def load_metadata(paths: dict):
    import pandas as pd
    df_a = pd.read_csv(paths["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns    = [c.lower().strip() for c in df_a.columns]
    rows = []
    for f in paths["set_b_dir"].iterdir():
        if f.suffix != ".wav":
            continue
        prefix = f.name.split("_")[0].lower()
        if prefix in VALID:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})
    df_b = pd.DataFrame(rows)
    df   = pd.concat([df_a, df_b], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan", "unlabeled", ""])]
    return df


def load_audio(fname: str, dataset: str, paths: dict) -> np.ndarray | None:
    folder = paths["set_a_dir"] if dataset == "A" else paths["set_b_dir"]
    path   = folder / Path(fname).name
    if not path.exists():
        return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target = SR * DURATION
        return np.pad(y, (0, max(0, target - len(y))))[:target]
    except Exception:
        return None

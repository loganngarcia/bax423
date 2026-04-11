#!/usr/bin/env python3
"""Patch UNSW_LightGBM.ipynb and RangeQueryText.ipynb in place."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "DataFiles"

UNSW_IMPORTS = r'''import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 808
TEST_SIZE = 0.2

_DATA = Path.cwd() / "DataFiles"
DATA_PATH = str(_DATA / "UNSW-NB15_1.csv")
FEATURE_PATH = str(_DATA / "NUSW-NB15_features.csv")
'''

UNSW_PREPROCESS = r'''df = df.copy()

# Categorical columns per UNSW-NB15 schema (nominal fields).
categorical_cols = [c for c in ["proto", "state", "service"] if c in df.columns]

# Normalize targets first.
df["attack_cat"] = (
    df["attack_cat"]
    .fillna("Normal")
    .astype(str)
    .str.strip()
    .replace({"": "Normal", "-": "Normal"})
)
df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)

# Drop leakage / identifiers from features.
drop_for_modeling = ["srcip", "dstip", "Stime", "Ltime"]
drop_for_modeling = [c for c in drop_for_modeling if c in df.columns]

X_raw = df.drop(columns=drop_for_modeling + ["Label", "attack_cat"])
y_binary = df["Label"]
y_multi_raw = df["attack_cat"]

# Coerce numeric columns (ports sometimes load as strings).
for c in X_raw.columns:
    if c not in categorical_cols:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
X_raw = X_raw.fillna(0)

# Log-transform heavy-tailed nonnegative numeric features.
num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    s = X_raw[c]
    if (s.min(skipna=True) or 0) >= 0:
        X_raw[c] = np.log1p(np.maximum(s, 0))

X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
print(f"Model feature matrix shape after one-hot encoding: {X.shape}")
'''

UNSW_RUN_BINARY = r'''def run_binary_experiment(model_name, model_kwargs):
    binary_imbalance_kwargs = {"is_unbalance": True}
    kw = dict(
        random_state=RANDOM_STATE,
        n_estimators=200,
        learning_rate=0.1,
        colsample_bytree=1.0,
        **binary_imbalance_kwargs,
        **model_kwargs,
    )
    kw.setdefault("max_depth", 6)
    kw.setdefault("subsample", 1.0)
    model = LGBMClassifier(**kw)

    t0 = time.perf_counter()
    model.fit(X_train_bin, y_train_bin)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(X_test_bin)
    infer_seconds = time.perf_counter() - t1

    infer_full_est_seconds = infer_seconds * (len(X) / len(X_test_bin))
    total_load_test_split = train_seconds + infer_seconds
    total_load_full_est = train_seconds + infer_full_est_seconds

    acc = accuracy_score(y_test_bin, preds)
    print(
        f"{model_name}: train={train_seconds:.4f}s, infer={infer_seconds:.4f}s, "
        f"total_load={total_load_test_split:.4f}s, acc={acc:.4f}"
    )

    return {
        "Model": model_name,
        "Train Time (s)": train_seconds,
        "Inference Time (s)": infer_seconds,
        "Total Load (s)": total_load_test_split,
        "Estimated Full Inference Load (s)": infer_full_est_seconds,
        "Estimated Full Total Load (s)": total_load_full_est,
        "Binary Accuracy": acc,
        "Imbalance Handling": "is_unbalance=True",
    }


binary_results = [xgb_result]

binary_results.append(
    run_binary_experiment("Vanilla LGBM", {"max_depth": 6})
)
binary_results.append(
    run_binary_experiment(
        "Deeper trees (num_leaves=127)",
        {"num_leaves": 127, "max_depth": 12},
    )
)
binary_results.append(
    run_binary_experiment(
        "GOSS boosting",
        {"boosting_type": "goss"},
    )
)
binary_results.append(
    run_binary_experiment(
        "Bagging 50% row subsample",
        {
            "subsample": 0.5,
            "subsample_freq": 1,
        },
    )
)
'''

UNSW_MULTICLASS = r'''label_encoder = LabelEncoder()
y_multi = label_encoder.fit_transform(y_multi_raw)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X,
    y_multi,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_multi,
)

multi_clf = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    n_jobs=-1,
)

t0 = time.perf_counter()
multi_clf.fit(X_train_multi, y_train_multi)
multi_train_seconds = time.perf_counter() - t0

t1 = time.perf_counter()
multi_preds = multi_clf.predict(X_test_multi)
multi_infer_seconds = time.perf_counter() - t1

multi_infer_full_est_seconds = multi_infer_seconds * (len(X) / len(X_test_multi))
multi_total_load_test_split = multi_train_seconds + multi_infer_seconds
multi_total_load_full_est = multi_train_seconds + multi_infer_full_est_seconds
multi_acc = accuracy_score(y_test_multi, multi_preds)

multi_metrics = {
    "Model": "Multiclass LGBM",
    "Train Time (s)": multi_train_seconds,
    "Inference Time (s)": multi_infer_seconds,
    "Total Load (s)": multi_total_load_test_split,
    "Estimated Full Inference Load (s)": multi_infer_full_est_seconds,
    "Estimated Full Total Load (s)": multi_total_load_full_est,
    "Multiclass Accuracy": multi_acc,
    "Imbalance Handling": "class_weight='balanced'",
    "Num Classes": len(label_encoder.classes_),
}

print("\nCompact class-wise report:")
print(classification_report(y_test_multi, multi_preds, target_names=label_encoder.classes_, zero_division=0))
'''


def patch_unsw() -> None:
    p = ROOT / "UNSW_LightGBM.ipynb"
    nb = json.loads(p.read_text())
    cells = nb["cells"]
    # cell 2
    cells[2]["source"] = [line + "\n" for line in UNSW_IMPORTS.strip().split("\n")]
    # cell 6
    cells[6]["source"] = [line + "\n" for line in UNSW_PREPROCESS.strip().split("\n")]
    # cell 11
    cells[11]["source"] = [line + "\n" for line in UNSW_RUN_BINARY.strip().split("\n")]
    # cell 17
    cells[17]["source"] = [line + "\n" for line in UNSW_MULTICLASS.strip().split("\n")]
    p.write_text(json.dumps(nb, indent=1))
    print("Patched", p)


RANGE_IMPORTS = r'''import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mmh3
import numpy as np
'''

RANGE_LOAD = r'''FRANKPATH = str(Path.cwd() / "DataFiles" / "Frankenstein.txt")
with open(FRANKPATH, "r", encoding="utf-8", errors="ignore") as f:
    frankenstein = f.read()

# Hyperparameter: words per interval.
interval_words = 4000

clean_text = frankenstein.lower()
clean_text = re.sub(r"[^a-z\s]", " ", clean_text)
tokens = re.findall(r"[a-z]+", clean_text)

intervals = [
    tokens[i : i + interval_words]
    for i in range(0, len(tokens), interval_words)
]

print(f"Total words: {len(tokens):,}")
print(f"Interval size (words): {interval_words:,}")
print(f"Number of intervals: {len(intervals):,}")
'''

# Cell 3 in Range notebook contains class + interval_sketches + range_query - we replace from interval_sketches onward after class (keep class from original or embed)

RANGE_SKETCH_BLOCK = r'''# Shared sketch accuracy hyperparameters.
epsilon = 0.005
delta = 0.01


interval_sketches = []
for interval in intervals:
    cms = CountMinSketch(epsilon=epsilon, delta=delta)
    for tok in interval:
        cms.update(tok)
    interval_sketches.append(cms)


def query_interval(interval_idx: int, word: str) -> int:
    if interval_idx < 0 or interval_idx >= len(interval_sketches):
        raise IndexError("interval_idx out of range")
    return interval_sketches[interval_idx].query(word.lower())


def range_query(word: str, start_interval: int, end_interval: int) -> int:
    w = word.lower()
    total = 0
    for idx in range(start_interval, end_interval + 1):
        total += query_interval(idx, w)
    return int(total)


for w in ["the", "monster"]:
    print(f"Total estimated count for '{w}': {range_query(w, 0, len(interval_sketches)-1)}")

# Word-index ranges (0-based): first 4000 words = interval 0 only.
# "fire" words 12000-16000 => interval 3 only (4000 per interval).
# "science" first 20000 words => intervals 0-4.
print("victor [0:4000] via intervals:", range_query("victor", 0, 0))
print("fire [12000:16000] via intervals:", range_query("fire", 3, 3))
print("science [0:20000] via intervals:", range_query("science", 0, 4))
'''

RANGE_DYADIC = r'''def build_dyadic_intervals(tokens, epsilon, delta):
    n = len(tokens)
    dyadic = {}
    if n == 0:
        return dyadic, n
    max_level = int(math.floor(math.log2(n)))
    for level in range(0, max_level + 1):
        block_len = 1 << level
        print("Level: " + str(level) + ", Interval Lengths: " + str(block_len))
        for start_word in range(0, n, block_len):
            end_word = start_word + block_len
            if end_word > n:
                continue
            cms = CountMinSketch(epsilon=epsilon, delta=delta)
            for token in tokens[start_word:end_word]:
                cms.update(token)
            dyadic[(level, start_word)] = cms
    return dyadic, n


# Use epsilon per sketch so that summing O(log n) dyadic blocks keeps additive error ~5% of interval mass.
RQEpsilon = 0.05 / (2 * max(1, int(np.ceil(np.log2(len(tokens))))))

dyadic_sketches, n_tokens = build_dyadic_intervals(
    tokens=tokens,
    epsilon=RQEpsilon,
    delta=delta,
)


def dyadic_range_query(word: str, start_word_idx: int, end_word_idx: int):
    if start_word_idx < 0 or end_word_idx >= n_tokens or start_word_idx > end_word_idx:
        raise ValueError("Invalid word-index range")

    i = start_word_idx
    segments = []

    while i <= end_word_idx:
        max_k = int(math.floor(math.log2(end_word_idx - i + 1)))
        k = max_k
        while k >= 0:
            block_len = 1 << k
            if (i % block_len == 0) and (i + block_len - 1 <= end_word_idx) and ((k, i) in dyadic_sketches):
                segments.append((k, i, block_len))
                i += block_len
                break
            k -= 1
        if k < 0:
            raise RuntimeError("Could not decompose range into dyadic blocks")

    estimate = sum(dyadic_sketches[(k, s)].query(word.lower()) for k, s, _ in segments)
    return {
        "word": word,
        "word_range": (start_word_idx, end_word_idx),
        "dyadic_segments": segments,
        "estimated_count": estimate,
    }


NFrankenstein = dyadic_range_query("frankenstein", 0, 9999)["estimated_count"]
NVictor = dyadic_range_query("victor", 39999, 60346)["estimated_count"]
NCreature = dyadic_range_query("creature", 28, 310)["estimated_count"]
print(NFrankenstein, NVictor, NCreature)
'''

RANGE_ANCHOR = r'''def estimate_from_anchor_positions(anchor_word: str, related_word: str, k: int = 20):
    if k < 0:
        raise ValueError("k must be non-negative")
    aw = anchor_word.lower()
    rw = related_word.lower()
    anchor_positions = [i for i, t in enumerate(tokens) if t == aw]
    n_tokens_local = len(tokens)
    per_anchor_estimates = []
    for idx in anchor_positions:
        start = min(idx + 1, n_tokens_local - 1)
        end = min(idx + k, n_tokens_local - 1)
        if start > end:
            est = 0
        else:
            est = dyadic_range_query(rw, start, end)["estimated_count"]
        per_anchor_estimates.append(
            {
                "anchor_idx": idx,
                "query_range": (start, end),
                "related_estimate": est,
            }
        )
    total_related_estimate = sum(item["related_estimate"] for item in per_anchor_estimates)
    nonzero = sum(1 for item in per_anchor_estimates if item["related_estimate"] > 0)
    return {
        "anchor_word": anchor_word,
        "related_word": related_word,
        "k": k,
        "num_anchor_positions": len(anchor_positions),
        "anchors_with_nonzero_related_estimate": nonzero,
        "total_related_estimate_over_all_anchor_windows": total_related_estimate,
        "per_anchor_estimates": per_anchor_estimates,
    }


ex1 = estimate_from_anchor_positions("frankenstein", "monster", k=20)
print({
    "anchor_word": ex1["anchor_word"],
    "related_word": ex1["related_word"],
    "k": ex1["k"],
    "num_anchor_positions": ex1["num_anchor_positions"],
    "anchors_with_nonzero_related_estimate": ex1["anchors_with_nonzero_related_estimate"],
    "total_related_estimate_over_all_anchor_windows": ex1["total_related_estimate_over_all_anchor_windows"],
})
print("First five per-anchor windows:", ex1["per_anchor_estimates"][:5])

ex2 = estimate_from_anchor_positions("victor", "creature", k=300)
print({
    "anchor_word": ex2["anchor_word"],
    "related_word": ex2["related_word"],
    "k": ex2["k"],
    "num_anchor_positions": ex2["num_anchor_positions"],
    "anchors_with_nonzero_related_estimate": ex2["anchors_with_nonzero_related_estimate"],
    "total_related_estimate_over_all_anchor_windows": ex2["total_related_estimate_over_all_anchor_windows"],
})
'''

RANGE_MATRIX = r'''from collections import Counter

top50 = [w for w, _ in Counter(tokens).most_common(50)]
anchors = top50
others = top50
mat = np.zeros((len(anchors), len(others)))
for i, a in enumerate(anchors):
    na = max(1, tokens.count(a))
    for j, o in enumerate(others):
        if i == j:
            continue
        tot = estimate_from_anchor_positions(a, o, k=100)["total_related_estimate_over_all_anchor_windows"]
        mat[i, j] = tot / (na * 100.0)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(others)))
ax.set_xticklabels(others, rotation=90, fontsize=6)
ax.set_yticks(range(len(anchors)))
ax.set_yticklabels(anchors, fontsize=6)
ax.set_title("Estimated co-occurrence rate in 100-word window after anchor (top-50 words)")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
'''

RANGE_RANDOM = r'''freq = Counter(tokens)
candidates = [w for w, c in freq.items() if c > 8]
rng = np.random.default_rng(808)
if len(candidates) < 70:
    words70 = candidates
else:
    words70 = list(rng.choice(candidates, size=70, replace=False))

anchors_r = words70
others_r = words70
mat_r = np.zeros((len(anchors_r), len(others_r)))
for i, a in enumerate(anchors_r):
    na = max(1, tokens.count(a))
    for j, o in enumerate(others_r):
        if i == j:
            continue
        tot = estimate_from_anchor_positions(a, o, k=100)["total_related_estimate_over_all_anchor_windows"]
        mat_r[i, j] = tot / (na * 100.0)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(mat_r, vmin=0, vmax=1, cmap="magma")
ax.set_title("Random 70 words (>8 occurrences) — co-occurrence proxy")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
'''


def patch_range() -> None:
    import re

    p = ROOT / "RangeQueryText.ipynb"
    nb = json.loads(p.read_text())
    c = nb["cells"]
    c[1]["source"] = [line + "\n" for line in RANGE_IMPORTS.strip().split("\n")]
    c[2]["source"] = [line + "\n" for line in RANGE_LOAD.strip().split("\n")]
    s3 = "".join(c[3]["source"])
    start = s3.find("class CountMinSketch")
    if start == -1:
        raise RuntimeError("CountMinSketch class not found in Range cell 3")
    tail = s3[start:]
    tail = re.sub(
        r"##TODO: Create countmin sketches that correspond to fixed intervals of words in the book\ninterval_sketches = \[\]\nfor interval in intervals:\n    None\n",
        "interval_sketches = []\nfor interval in intervals:\n    cms = CountMinSketch(epsilon=epsilon, delta=delta)\n    for tok in interval:\n        cms.update(tok)\n    interval_sketches.append(cms)\n\n",
        tail,
    )
    tail = re.sub(
        r"def range_query\(word: str, start_interval: int, end_interval: int\) -> int:\n  ##TODO: Define a function that naively returns the range query for given intervals\n  #by utilizing the query_interval function\.\n    None\n",
        "def range_query(word: str, start_interval: int, end_interval: int) -> int:\n    w = word.lower()\n    total = 0\n    for idx in range(start_interval, end_interval + 1):\n        total += query_interval(idx, w)\n    return int(total)\n\n",
        tail,
    )
    extra = (
        '\nprint("victor first 4000 words (interval 0):", range_query("victor", 0, 0))\n'
        'print("fire words 12001-16000 (interval 3):", range_query("fire", 3, 3))\n'
        'print("science first 20000 words (intervals 0-4):", range_query("science", 0, 4))\n\n'
    )
    idx = tail.find("# Basic sanity check")
    if idx != -1:
        tail = tail[:idx] + extra + tail[idx:]
    c[3]["source"] = [ln + "\n" for ln in tail.splitlines()]
    if c[3]["source"] and not c[3]["source"][-1].endswith("\n"):
        c[3]["source"][-1] += "\n"
    c[5]["source"] = [line + "\n" for line in RANGE_DYADIC.strip().split("\n")]
    c[6]["source"] = [line + "\n" for line in RANGE_ANCHOR.strip().split("\n")]
    c[7]["source"] = [line + "\n" for line in RANGE_MATRIX.strip().split("\n")]
    c[9]["source"] = [line + "\n" for line in RANGE_RANDOM.strip().split("\n")]
    p.write_text(json.dumps(nb, indent=1))
    print("Patched", p)


def main() -> None:
    patch_unsw()
    patch_range()


if __name__ == "__main__":
    main()

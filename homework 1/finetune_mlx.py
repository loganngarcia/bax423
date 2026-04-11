#!/usr/bin/env python3
"""
BAX 423 HW1 — Part 2 — Apple Silicon fast path (MLX + BERT-base).

PyTorch + MPS on Mac is often **~1–2 steps/s** on this assignment (many hours for 3 epochs).
This script uses **Apple MLX** with **bert-base-uncased** (same uncased WordPiece tokenizer
family as DistilBERT; backbone is full BERT). Typical MLX throughput here is **~5–15+ steps/s**
on recent M-series chips, so expect on the order of **~20–60 minutes** for 3 epochs instead of
**~3–6+ hours** with PyTorch+MPS — exact time depends on RAM and chip tier.

**Weights (Hugging Face Hub, not a manual GitHub clone):** pretrained MLX `weights.npz` are loaded
from **`mlx-community/bert-base-uncased-mlx`** (see https://huggingface.co/mlx-community/bert-base-uncased-mlx ).
That repo is the best match for this repo’s `mlx_bert/model.py` (mlx-examples–style BERT). Hub search
also lists `mlx-community/bert-large-uncased-mlx` (larger/slower) and encoder-adjacent MLX repos; a
**DistilBERT** classification checkpoint such as `Mlxa/atd-distilbert` uses **safetensors + Transformers**,
not this raw `weights.npz` layout — use **`finetune.py`** on GPU/Colab for the assignment’s DistilBERT
starter if you need that exact architecture.

Keep `finetune.py` for Colab/CUDA submission if the grader expects the exact DistilBERT starter;
submit this log + notebook as your “local fast” run.

Usage:
  cd "homework 1" && source .venv/bin/activate
  pip install mlx datasets transformers numpy scikit-learn huggingface_hub
  python finetune_mlx.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx.optimizers import AdamW
from sklearn.metrics import accuracy_score, classification_report, f1_score
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mlx_bert.model import Bert  # noqa: E402

BERT_NAME = "bert-base-uncased"
# HF Hub: mlx-community — official MLX BERT-base weights (npz), compatible with mlx_bert/model.py
HF_MLX_REPO = "mlx-community/bert-base-uncased-mlx"
MAX_LEN = 225
BATCH_SIZE = 48
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
SEED = 808


class BertForSequenceClassificationMLX(nn.Module):
    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        self.bert = Bert(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        token_type_ids: mx.array | None = None,
    ) -> mx.array:
        _, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        return self.classifier(self.dropout(pooled))


def resolve_mlx_weights_npz() -> str:
    """Download `weights.npz` from Hugging Face Hub (preferred) or build via mlx_bert/convert.py."""
    fallback = _ROOT / "mlx_bert" / "weights" / "bert-base-uncased.npz"
    if fallback.exists():
        return str(fallback)
    try:
        path = hf_hub_download(
            repo_id=HF_MLX_REPO,
            filename="weights.npz",
        )
        print(f"Using MLX BERT weights from Hugging Face Hub: {HF_MLX_REPO} → {path}")
        return path
    except Exception as e:
        print(f"Hub download failed ({e}); falling back to local PyTorch→npz conversion…")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(
            [
                sys.executable,
                str(_ROOT / "mlx_bert" / "convert.py"),
                "--bert-model",
                BERT_NAME,
                "--mlx-model",
                str(fallback),
            ],
            check=True,
        )
        return str(fallback)


def main() -> None:
    mx.random.seed(SEED)
    np.random.seed(SEED)

    weights_path = resolve_mlx_weights_npz()

    print("Loading Newswire dataset…")
    ds = load_dataset(
        "dell-research-harvard/newswire",
        data_files=[
            "1965_data_clean.json",
            "1966_data_clean.json",
            "1967_data_clean.json",
            "1968_data_clean.json",
        ],
    )
    split = ds["train"].train_test_split(test_size=0.2, seed=SEED)
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    def tokenize_split(split_name: str):
        d = split[split_name]

        def _tok(batch):
            t = tokenizer(
                batch["article"],
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="np",
            )
            return {
                "input_ids": t["input_ids"],
                "attention_mask": t["attention_mask"],
                "labels": batch["civil_rights"],
            }

        return d.map(
            _tok,
            batched=True,
            remove_columns=d.column_names,
            desc=f"Tokenizing {split_name}",
        )

    train_ds = tokenize_split("train")
    test_ds = tokenize_split("test")

    train_ids = np.asarray(train_ds["input_ids"], dtype=np.int32)
    train_mask = np.asarray(train_ds["attention_mask"], dtype=np.float32)
    train_y = np.asarray(train_ds["labels"], dtype=np.int32)

    test_ids = np.asarray(test_ds["input_ids"], dtype=np.int32)
    test_mask = np.asarray(test_ds["attention_mask"], dtype=np.float32)
    test_y = np.asarray(test_ds["labels"], dtype=np.int32)

    config = AutoConfig.from_pretrained(BERT_NAME)
    model = BertForSequenceClassificationMLX(config, num_labels=2)
    model.bert.load_weights(weights_path)

    optimizer = AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    n_train = len(train_y)
    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS

    print(
        f"MLX train | n={n_train} batch={BATCH_SIZE} epochs={EPOCHS} → ~{total_steps} optimizer steps",
        flush=True,
    )

    idx = np.arange(n_train)
    global_step = 0
    t0 = time.perf_counter()

    model.train()
    for epoch in range(EPOCHS):
        np.random.shuffle(idx)
        ep_loss = 0.0
        nb = 0

        for start in range(0, n_train, BATCH_SIZE):
            sel = idx[start : start + BATCH_SIZE]
            input_ids = mx.array(train_ids[sel])
            attention_mask = mx.array(train_mask[sel])
            labels = mx.array(train_y[sel])

            def loss_fn():
                logits = model(input_ids, attention_mask, None)
                return mx.mean(nn.losses.cross_entropy(logits, labels))

            loss, grads = nn.value_and_grad(model, loss_fn)()
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            ep_loss += float(loss)
            nb += 1
            global_step += 1

            if global_step % 300 == 0:
                dt = time.perf_counter() - t0
                sps = global_step / max(dt, 1e-6)
                print(
                    f"  step {global_step}/{total_steps}  loss={float(loss):.4f}  "
                    f"~{sps:.1f} step/s  {dt/60:.1f} min elapsed"
                )

        print(f"Epoch {epoch + 1}/{EPOCHS} — mean batch loss: {ep_loss / max(1, nb):.4f}")

    wall = time.perf_counter() - t0
    print(f"\nTraining wall time: {wall/60:.1f} min ({wall:.0f} s)")

    # Evaluation
    model.eval()
    all_preds: list[int] = []
    n_test = len(test_y)
    for start in range(0, n_test, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n_test))
        logits = model(
            mx.array(test_ids[sl]),
            mx.array(test_mask[sl]),
            None,
        )
        mx.eval(logits)
        preds = np.argmax(np.asarray(logits), axis=-1)
        all_preds.extend(preds.tolist())

    acc = accuracy_score(test_y, all_preds)
    f1p = f1_score(test_y, all_preds, pos_label=1, average="binary")
    print(f"\nTest accuracy: {acc:.4f}  |  F1 (label=1): {f1p:.4f}")
    print(classification_report(test_y, all_preds, digits=4))

    # Perplexity-style scalar from mean CE (same trick as PyTorch script)
    # Approximate eval loss: average -log p(true class)
    ce_sum = 0.0
    for start in range(0, n_test, BATCH_SIZE):
        sl = slice(start, min(start + BATCH_SIZE, n_test))
        logits = model(mx.array(test_ids[sl]), mx.array(test_mask[sl]), None)
        mx.eval(logits)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        yb = mx.array(test_y[sl])
        ce = -mx.take_along_axis(log_probs, mx.expand_dims(yb, -1), axis=-1).squeeze(-1)
        ce_sum += float(mx.sum(ce))
    mean_ce = ce_sum / n_test
    print(f"mean CE (eval): {mean_ce:.4f}")
    print(f"exp(mean CE) (perplexity-style): {math.exp(mean_ce):.4f}")

    if acc < 0.93:
        print(
            "\nIf accuracy is below 93%, increase EPOCHS to 4 or LR to 3e-5, "
            "or use the PyTorch `finetune.py` on a GPU runtime."
        )

    # Rubric Part 2(e): resource usage (best effort on macOS; no NVIDIA GPU on Apple Silicon)
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        print(f"Peak RSS (ru_maxrss, OS-dependent units): {ru.ru_maxrss}")
    except Exception as e:
        print("Could not read RSS:", e)
    print("(GPU usage: N/A on typical Mac — use Colab/CUDA log from finetune.py if required.)")


if __name__ == "__main__":
    main()

# BAX 423 — Homework 1 — Part 2 — Modeling & data decisions (for PDF item 2d)

**Group:** Logan Garcia, Bonnie Hines  

Fill in the bracketed numbers from your training log (`finetune.py` on Colab/GPU or `finetune_mlx.py` on Mac).

## Decisions

1. **Encoder:** Course starter uses **DistilBERT** (`distilbert-base-uncased`) in `finetune.py`. DistilBERT retains much of BERT’s accuracy with fewer layers — good for historical news text and GPU memory.

2. **Sequence length 225:** Matches the assignment cap; truncates very long OCR articles without padding the entire corpus to 512.

3. **Whitespace cleanup:** Collapsing repeated spaces reduces noise from scanning/layout without aggressive stemming that could hurt rare civil-rights terms.

4. **Train/validation split:** 80/20 with fixed seed (808) for reproducibility.

5. **Metrics:** Accuracy for the overall rubric target (≥93%); **precision/recall/F1 on the positive (civil rights) class** because the label distribution is imbalanced — success on the minority class matters for this task.

6. **Optimization:** AdamW with weight decay, warmup, a few epochs — standard GLUE-style fine-tuning for BERT-family models.

## Results (paste from your run)

| Field | Value |
|--------|--------|
| Best / final **test accuracy** | _paste_ |
| **eval_loss** (cross-entropy) | _paste_ |
| **exp(eval_loss)** (perplexity-style scalar) | _paste_ |
| **Peak CPU** (e.g. Activity Monitor / `ru_maxrss`) | _paste_ |
| **Peak GPU** (nvidia-smi / Colab) | _paste from GPU run_ |
| **Wall-clock training time** | _paste_ |

**Note:** For Apple Silicon MLX (`finetune_mlx.py`), use **bert-base-uncased** weights from Hugging Face `mlx-community/bert-base-uncased-mlx` for speed; for strict DistilBERT parity, run `finetune.py` on a **GPU** and paste those numbers here.

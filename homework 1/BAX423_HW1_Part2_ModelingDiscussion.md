# Part 2 — Modeling and data choices

**Logan Garcia, Bonnie Hines**

## Modeling decisions

We fine-tuned a transformer encoder for binary classification on the Newswire civil-rights label. The handout starter uses **DistilBERT** (`distilbert-base-uncased`) in `finetune.py`: fewer layers than full BERT, with similar behavior on text classification and reasonable GPU memory use.

We used **max length 225** tokens to match the assignment cap and avoid padding every example to 512. Training data were split **80/20** train/validation with seed **808**. We report **accuracy** on the held-out test set (rubric asks for at least **93%**) and also look at **precision, recall, and F1 on the positive (civil rights) class** because the classes are not perfectly balanced. The handout also asks for a **perplexity-style number**: we record **exp(eval loss)** from the evaluation output (same scalar printed as “exp(eval_loss)” / “perplexity analog” in `finetune.py`). Optimization follows a standard setup: **AdamW**, weight decay, and a small number of epochs with learning rate per the starter.

For local runs on Apple Silicon we used an optional **MLX** script (`finetune_mlx.py`) with **BERT-base** weights from the Hugging Face Hub; the starter DistilBERT run should be taken from `finetune.py` on a **GPU** when comparing to the official notebook.

## Results

Numbers below come from the training run output (GPU run of `finetune.py` and/or Mac run of `finetune_mlx.py`).

| Metric | Value |
|--------|--------|
| Test accuracy | |
| Cross-entropy (eval) | |
| exp(cross-entropy) | |
| Peak CPU memory (e.g. `ru_maxrss` on Mac) | |
| Peak GPU memory (if applicable) | |
| Training time (wall clock) | |

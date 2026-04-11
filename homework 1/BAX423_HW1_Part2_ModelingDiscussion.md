# Part 2 Modeling and data choices

Logan Garcia, Bonnie Hines

## Decisions

We fine-tuned a transformer encoder for binary classification on the Newswire civil-rights label. The starter uses DistilBERT `distilbert-base-uncased` in `finetune.py`. It has fewer layers than full BERT with similar behavior on text classification and lighter GPU memory than large BERT.

We used max length 225 tokens to match the assignment cap and avoid padding every row to 512. Train and validation split 80 over 20 with seed 808. We care about accuracy on the held-out test set because the rubric wants about 93% or better. We also report precision, recall, and F1 on the positive civil rights class because positives are rarer. The handout asks for a perplexity-style scalar. We use exp of eval loss from the trainer output, same line as in `finetune.py` printed as exp eval loss or perplexity-style analog. Optimization is AdamW, weight decay, a few epochs, learning rate from the starter.

For optional local runs on Apple Silicon we have `finetune_mlx.py` with BERT-base weights from Hugging Face. The graded DistilBERT path should come from `finetune.py` on GPU when you compare to the course notebook.

## Results

Fill the **`finetune.py` Colab GPU** row after that run. Optional MLX numbers are logged in **`BAX423_HW1_Part2_MLX_results.md`** from the completed `--fast` local run.

### Required: `finetune.py` on Colab GPU

| Metric | Value |
|--------|--------|
| Test accuracy | |
| Cross-entropy eval | |
| exp cross-entropy | |
| Peak CPU memory ru_maxrss if printed | |
| Peak GPU memory if printed | |
| Training time wall clock | |

### Optional: local MLX `finetune_mlx.py --fast` Mac run

| Metric | Value |
|--------|--------|
| Test accuracy | 0.9686 |
| Training wall time | 34.7 min |
| F1 civil rights label 1 | 0.7103 |

Details in **`BAX423_HW1_Part2_MLX_results.md`**. After the next MLX run, **`part2_mlx_metrics.json`** will list mean CE, exp mean CE, and RSS automatically.

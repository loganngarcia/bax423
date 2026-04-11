# Part 2 optional local MLX run log

Logan Garcia, Bonnie Hines

This file records one completed **`finetune_mlx.py --fast`** run on Apple Silicon. The handout still expects **`finetune.py`** with DistilBERT on a GPU for the main submission. Treat this as a supplemental local benchmark, not a replacement for the Colab run.

## Run settings

| Setting | Value |
|---------|--------|
| Script | `finetune_mlx.py --fast` |
| Weights | `mlx-community/bert-base-uncased-mlx` |
| Training rows | 20,000 of 114,442 available |
| Optimizer steps | 834 total, batch 48, 2 epochs, lr `3e-5` |
| Epoch 1 mean batch loss | 0.1479 |
| Epoch 2 mean batch loss | 0.1284 |

## Metrics from this run

| Metric | Value |
|--------|--------|
| Test accuracy | **0.9686** about **96.86%** |
| F1 civil rights label 1 | **0.7103** |
| Precision label 1 | 0.8175 |
| Recall label 1 | 0.6279 |
| Training wall time | **34.7 min** 2084 s |
| Test set size | 28,611 rows |

**Mean CE, exp mean CE, and peak RSS** for the next MLX run are written automatically to **`part2_mlx_metrics.json`** in the same folder as `finetune_mlx.py`. That file did not exist for the run logged above. Re-run once if you want those three numbers on disk, or copy them from your terminal output if you still have it.

## One-line summary

Local MLX fast path reached **96.86%** test accuracy on the subset run, above the **93%** bar for accuracy, with about **35 minutes** training time.

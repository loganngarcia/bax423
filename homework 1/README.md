# BAX 423 — Homework 1

**Course:** BAX 423  
**Group:** Logan Garcia, Bonnie Hines  

This folder holds our **Homework 1** submission materials: Part 1 (ML design write-up), Part 2 (Newswire fine-tuning + notebook), Part 3 (attention mechanisms notebook), and Part 4 (NOAA tornado analysis with HTML + PDF report).

---

## Layout

| Path | What it is |
|------|------------|
| `finetune.py` | **PyTorch + Hugging Face `Trainer`** — DistilBERT, best on **GPU / Colab** |
| `finetune_mlx.py` | **Apple MLX** path — BERT-base weights from Hugging Face Hub |
| `BAX423_HW1_Part1_WriteUp.md` | Part 1 answers |
| `BAX423_HW1_Part2_FineTuning.ipynb` | Part 2 notebook (run after training) |
| `BAX423_HW1_Part3_AttentionMechanisms.ipynb` | Part 3 notebook |
| `part4/tornado_analysis.py` | Part 4 pipeline |
| `part4/output/` | Generated `tornado_report.html` and `tornado_report.pdf` |
| `requirements.txt` | Python dependencies |
| `BAX423_HW1_GRADING_CHECKLIST.txt` | Checklist vs. rubric |

---

## Quick start (local)

```bash
cd "homework 1"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Part 2 — PyTorch (DistilBERT, GPU recommended)

```bash
python finetune.py
```

Use **Google Colab** or a CUDA machine if you need speed; on Mac, PyTorch **MPS** can be slow for this workload.

### Part 2 — MLX (Apple Silicon)

```bash
python finetune_mlx.py
# Deadline / slow Mac (~0.3 step/s): ~under 1h wall time
PYTHONUNBUFFERED=1 python finetune_mlx.py --fast
```

Weights load from Hugging Face: [`mlx-community/bert-base-uncased-mlx`](https://huggingface.co/mlx-community/bert-base-uncased-mlx) (`weights.npz`).

### Part 4 — Tornado analysis + reports

```bash
cd part4
python tornado_analysis.py
```

Outputs go to `part4/output/` (HTML + PDF). Group names are set in `tornado_analysis.py`.

---

## AI tools & prompts (transparency)

Our instructor asked for clarity on how we used AI. Here is what we actually did:

1. **Cursor (coding assistant)**  
   Used for wiring scripts, debugging `Trainer` API changes (`processing_class` vs. deprecated `tokenizer=`), MLX training loops, ReportLab PDF/HTML for Part 4, and keeping the repo organized for submission.

2. **Hugging Face Hub / “Hugging Face MCP” in Cursor**  
   We asked the assistant to **find MLX-compatible BERT checkpoints on Hugging Face** so we could avoid a painfully slow **PyTorch + MPS** fine-tune on Mac. The assistant pointed us to **`mlx-community/bert-base-uncased-mlx`**, which matches the vendored `mlx_bert/model.py` layout (`weights.npz`).

3. **Why MLX helped**  
   In our setup, MPS was profiling around **~1–2 steps/s** for this assignment-style run (on the order of **many hours** for three epochs). MLX on Apple Silicon typically reaches **~5–15+ steps/s** for this pattern on capable hardware; our laptop was slower (**~0.3 step/s**), so we also used **`--fast`** (subset + fewer epochs) when needed. *Exact wall times belong in local `part2_mlx_run*.log` files after a run.*  
   *Optional sanity check:* a **tiny smoke run** (few steps) can finish in **~2 minutes**; **full rubric training** uses the script defaults and takes longer.

4. **What we did *not* outsource**  
   Course reasoning in the write-ups, interpreting attention outputs, and the business / policy narrative for Part 1 and Part 4 are our own, edited after drafting help.

---

## Status / what’s left before Canvas

See `BAX423_HW1_GRADING_CHECKLIST.txt`. Typical remaining items:

- **Part 2:** ~3 min **screen recording** (`.mp4`) — required by rubric.  
- **Part 2 & 3:** Notebooks **executed end-to-end** with outputs saved.  
- **Part 2:** Final **test accuracy ≥ 0.93** logged; paste metrics into `BAX423_HW1_Part2_ModelingDiscussion.md` / notebook.  
- **Part 4(e):** Email from a **non-UMN Gmail** (template in `part4/`).  

---

## License / course use

This repo is for **educational submission** in BAX 423. Do not misrepresent authorship; follow your course AI policy.

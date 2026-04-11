# Homework 1 — BAX 423

**Logan Garcia** · **Bonnie Hines**

This folder is everything we’re turning in for Homework 1: short written answers, two notebooks, fine-tuning code for the Newswire task, and a small NOAA tornado analysis with HTML and PDF outputs.

---

## What each part is

**Part 1 — Design write-up**  
Answers to the ML design questions in `BAX423_HW1_Part1_WriteUp.md` (export to PDF too if the course asks for it).

**Part 2 — Newswire classification**  
Fine-tune a transformer on the Harvard Newswire dataset, hit at least **93% test accuracy**, and document choices plus metrics. Starter script: `finetune.py` (DistilBERT, matches the assignment handout). There is also `finetune_mlx.py` for Apple Silicon if you want to train locally with MLX; use Colab + GPU if PyTorch on your Mac is too slow.

Supporting files: `BAX423_HW1_Part2_FineTuning.ipynb`, `BAX423_HW1_Part2_ModelingDiscussion.md`.

**Part 3 — Attention**  
`BAX423_HW1_Part3_AttentionMechanisms.ipynb` — complete the TODOs, answer the intuition questions, and make sure **every cell has been run** so outputs show when you submit.

**Part 4 — Tornado analysis**  
`part4/tornado_analysis.py` pulls NOAA tornado data (EF2+, 2020–2025), runs the analyses the handout asks for, and writes `part4/output/tornado_report.html` and `tornado_report.pdf` with **both** of our names on the PDF. There is an email template in `part4/` for Part 4(e); send from a **personal Gmail**, not your school address, with the subject line and wording the assignment specifies.

---

## Setup once

```bash
cd "homework 1"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Commands we actually use

**Fine-tuning (recommended for speed)**  
Use **Google Colab with a GPU** and run `finetune.py` there if your laptop is slow.

**Fine-tuning on a Mac (MLX)**  
From this folder, with the venv active:

```bash
python finetune_mlx.py
```

If a full run would take too long, there is a shorter schedule:

```bash
PYTHONUNBUFFERED=1 python finetune_mlx.py --fast
```

Pretrained MLX weights come from Hugging Face: [`mlx-community/bert-base-uncased-mlx`](https://huggingface.co/mlx-community/bert-base-uncased-mlx).

**Part 4 reports**

```bash
cd part4
python tornado_analysis.py
```

Group / Canvas name can be edited at the top of `tornado_analysis.py` if needed.

---

## AI use (course transparency)

We used **Cursor** to help wire up training scripts, fix library API changes (for example Hugging Face `Trainer` arguments), build the Part 4 HTML/PDF, and keep the repo tidy. We also used **Hugging Face Hub** to pick an MLX-compatible BERT checkpoint (`mlx-community/bert-base-uncased-mlx`) so local training was feasible. Judgment calls in the write-ups, interpretation of attention plots, and the substance of the business-facing answers are ours—we edited anything the tools drafted so it reflects what we actually think.

---

## Before Canvas

Double-check against the official handout and rubric; in short:

- Part 2: **test accuracy ≥ 0.93**, metrics (including something like perplexity / exp(loss) where asked), and the **short screen recording** (~3 minutes).  
- Both notebooks: **all cells executed**, outputs visible.  
- Part 4: **PDF and HTML** with names, plus the **email** from a non-UMN Gmail if that part is required.

A tighter item-by-item list lives in `BAX423_HW1_GRADING_CHECKLIST.txt` if you want a literal checklist.

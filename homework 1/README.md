# BAX 423 — Homework 1

**Students:** Logan Garcia, Bonnie Hines  

This directory contains our submission for Homework 1. Below is what each file or folder is for.

---

## Contents

| Item | Description |
|------|-------------|
| `HW1 (2).pdf` | Assignment prompt (course copy). |
| `HW1 Rubric.xlsx` | Grading rubric. |
| `BAX423_HW1_Part1_WriteUp.md` | Part 1 written answers. |
| `BAX423_HW1_Part1_WriteUp.pdf` | Part 1 export (if required on Canvas). |
| `finetune.py` | Part 2: DistilBERT fine-tuning with Hugging Face `Trainer` (intended for GPU / Colab). |
| `finetune_mlx.py` | Part 2 (optional): same task with MLX on Apple Silicon; weights from Hugging Face Hub. |
| `mlx_bert/` | Helper module used by `finetune_mlx.py`. |
| `BAX423_HW1_Part2_FineTuning.ipynb` | Part 2 notebook. |
| `BAX423_HW1_Part2_ModelingDiscussion.md` | Part 2 modeling discussion and results table. |
| `BAX423_HW1_Part3_AttentionMechanisms.ipynb` | Part 3 notebook. |
| `part4/tornado_analysis.py` | Part 4 script (NOAA analysis, HTML and PDF output). |
| `part4/output/` | Generated `tornado_report.html`, `tornado_report.pdf`, and `summary.json`. |
| `part4/email_template_part4e.txt` | Draft for the Part 4(e) email (personal Gmail only). |
| `requirements.txt` | Python dependencies. |

---

## Environment

```text
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Omit the `.venv` folder from your Canvas upload if the zip is too large; graders only need the files above and can reinstall packages from `requirements.txt`.

---

## Running the code (short)

- **Part 2 (GPU):** run `finetune.py` in Colab or another machine with CUDA.  
- **Part 2 (Mac, MLX):** from this directory, `python finetune_mlx.py` or `python finetune_mlx.py --fast` for a shorter local run (see script help).  
- **Part 4:** `cd part4` then `python tornado_analysis.py`.

---

## Course policy on tools

Where the syllabus asks for it, we have documented use of editors, libraries, and assistive software in the course’s required format. Substantive analysis in the write-ups and notebooks is our own.

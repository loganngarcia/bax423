# BAX 423 Homework 1

**Logan Garcia, Bonnie Hines**

## Contents

- Part 1: [BAX423_HW1_Part1_WriteUp.md](BAX423_HW1_Part1_WriteUp.md), [BAX423_HW1_Part1_WriteUp.pdf](BAX423_HW1_Part1_WriteUp.pdf)
- Part 2: [finetune.py](finetune.py), [BAX423_HW1_Part2_FineTuning.ipynb](BAX423_HW1_Part2_FineTuning.ipynb), [BAX423_HW1_Part2_ModelingDiscussion.md](BAX423_HW1_Part2_ModelingDiscussion.md), [part2_colab_metrics.json](part2_colab_metrics.json) (Colab DistilBERT run), [BAX423_HW1_Part2_MLX_results.md](BAX423_HW1_Part2_MLX_results.md) (MLX-only side run)
- Part 3: [BAX423_HW1_Part3_AttentionMechanisms.ipynb](BAX423_HW1_Part3_AttentionMechanisms.ipynb)
- Part 4: [part4/tornado_analysis.py](part4/tornado_analysis.py), [part4/output/](part4/output/), [part4/email_template_part4e.txt](part4/email_template_part4e.txt)

## Part 2 compute

We have Google Colab Pro. The DistilBERT runs that go with [finetune.py](finetune.py) and [BAX423_HW1_Part2_FineTuning.ipynb](BAX423_HW1_Part2_FineTuning.ipynb) were executed on Google Colab on an NVIDIA A100 GPU.

We added [finetune_mlx.py](finetune_mlx.py) and [mlx_bert/](mlx_bert/) as an optional local path for Macs with Apple Silicon (M1 and newer). From experience we know MLX is fast on-device there for this kind of work, so we used it for local runs and sanity checks. It is not a substitute for the required Colab DistilBERT stack; [BAX423_HW1_Part2_MLX_results.md](BAX423_HW1_Part2_MLX_results.md) is only for that MLX side.

Optional HF token cell in the Part 2 notebook for Hub downloads. Part 4 tie-breaking and filters follow the handout.

## Rerun

- `pip install -r requirements.txt` ([requirements.txt](requirements.txt))
- Colab: upload [finetune.py](finetune.py) and [BAX423_HW1_Part2_FineTuning.ipynb](BAX423_HW1_Part2_FineTuning.ipynb), GPU runtime, A100 if available on your account
- Part 4: `cd part4`, then `python tornado_analysis.py` ([part4/tornado_analysis.py](part4/tornado_analysis.py))
- MLX (optional, Apple Silicon Mac M1+): `python finetune_mlx.py` or `python finetune_mlx.py --fast` ([finetune_mlx.py](finetune_mlx.py))

Zip: omit `.venv` if the archive is oversized.

## AI usage

We used Cursor only (drafting, refactoring, debugging, and some Part 4 boilerplate). We ran training and notebooks ourselves, checked work against the rubric, and edited what we submitted. Models and data are from course materials and Hugging Face. The MLX scripts and the choice to use MLX were ours from prior experience with the framework, not a suggestion we followed from Cursor.

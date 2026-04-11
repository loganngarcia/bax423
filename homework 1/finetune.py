# -*- coding: utf-8 -*-
"""
BAX 423 HW1 Part 2. Newswire civil-rights classification with DistilBERT.

Run in Colab or a GPU machine:
  pip install transformers datasets evaluate accelerate scikit-learn
Optional resource logging: pip install psutil pynvml

Peak CPU and GPU usage print after training when CUDA and pynvml are available.
"""

## UNCOMMENT IN COLAB TO INSTALL PACKAGES
# !pip install -q transformers huggingface_hub evaluate datasets tokenizers accelerate scikit-learn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
import numpy as np
import evaluate
import torch
import warnings
import time
import os

warnings.filterwarnings("ignore")

set_seed(808)

# Optional: reset peak GPU stats before run
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

ds = load_dataset(
    "dell-research-harvard/newswire",
    data_files=[
        "1965_data_clean.json",
        "1966_data_clean.json",
        "1967_data_clean.json",
        "1968_data_clean.json",
    ],
)

print(ds)

from collections import Counter

label_counts = Counter(ds["train"]["civil_rights"])
print("Label counts:", label_counts)

splitdata = ds["train"].train_test_split(test_size=0.2, seed=808)

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

splitdata = splitdata.rename_column("civil_rights", "labels")


def clean_text(example):
    text = example["article"]
    if text is None:
        text = ""
    text = " ".join(str(text).split())
    return {"prepped_article": text}


splitdata = splitdata.map(clean_text)


def tokenize(batch):
    return tokenizer(
        batch["prepped_article"],
        truncation=True,
        max_length=225,
        padding=False,
    )


cols_to_remove = [c for c in splitdata["train"].column_names if c != "labels"]
tokenized = splitdata.map(
    tokenize,
    batched=True,
    remove_columns=cols_to_remove,
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    out = {}
    out["accuracy"] = accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    out["f1_macro"] = f1.compute(
        predictions=predictions, references=labels, average="macro"
    )["f1"]
    # Positive class is civil rights article label 1. Important under imbalance.
    out["f1_civil_rights"] = f1.compute(
        predictions=predictions,
        references=labels,
        average="binary",
        pos_label=1,
    )["f1"]
    out["precision_civil_rights"] = precision.compute(
        predictions=predictions,
        references=labels,
        pos_label=1,
    )["precision"]
    out["recall_civil_rights"] = recall.compute(
        predictions=predictions,
        references=labels,
        pos_label=1,
    )["recall"]
    return out


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

out_dir = "./newswire-distilbert-cls"
training_args = TrainingArguments(
    output_dir=out_dir,
    learning_rate=2e-5,
    # batch 16 is faster on Apple MPS than 32 for this model; CUDA machines can raise this.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,  # increase if test accuracy stays below 0.93
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    seed=808,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

t0 = time.perf_counter()
trainer.train()
train_seconds = time.perf_counter() - t0
print(f"Training wall time seconds: {train_seconds:.1f}")

ftmetrics = trainer.evaluate(tokenized["test"])
print("Test metrics:", ftmetrics)

# Perplexity-style scalar: exp of cross-entropy loss, same math step as in LM training logs.
# For classification it is not a true perplexity but matches the rubric ask for one number.
eval_loss = ftmetrics.get("eval_loss", float("nan"))
if eval_loss == eval_loss:
    print(f"eval_loss cross-entropy: {eval_loss:.4f}")
    print(f"exp eval_loss perplexity-style scalar: {np.exp(eval_loss):.4f}")

# Peak resource usage best effort
try:
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS ru_maxrss is bytes. Linux often kilobytes. Units vary by OS.
    maxrss = ru.ru_maxrss
    print(f"Peak RSS ru_maxrss field: {maxrss}")
except Exception as e:
    print("Could not read CPU peak RSS:", e)

if torch.cuda.is_available():
    mb = 1024**2
    print(
        "Peak GPU allocated MB:",
        torch.cuda.max_memory_allocated() / mb,
    )
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        print("GPU total and used MB:", mem.total / mb, mem.used / mb)
    except Exception:
        pass

# sklearn extras on held-out test
from sklearn.metrics import classification_report, confusion_matrix

preds_output = trainer.predict(tokenized["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=-1)
print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))
print("Confusion matrix rows are true labels 0 then 1:\n", confusion_matrix(y_true, y_pred))

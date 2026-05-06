import wandb
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import classification_report

wandb.init(project="socratic-engine")

dataset = load_from_disk("data/processed/")

labels = ["missing_concept", "misconception", "incomplete", "correct"]
label2id = {l: i for i, l in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

def preprocess(ex):
    text = f"Question: {ex['question']} [SEP] Correct answer: {ex['correct_answer']} [SEP] Student said: {ex['student_answer']} [SEP] Similarity: {ex['gap_score']:.2f}"
    
    tok = tokenizer(
        text,
        truncation=True,
        padding="max_length",   # ✅ IMPORTANT FIX
        max_length=256          # safe length
    )

    tok["labels"] = label2id[ex["gap_type"]]
    return tok

dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=4
)

args = TrainingArguments(
    output_dir="./models/gap_classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_dir="./logs",
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

trainer.train()

# preds = trainer.predict(dataset["test"])
# y_pred = preds.predictions.argmax(axis=1)
# y_true = dataset["test"]["gap_type"]

# print(classification_report(y_true, y_pred))

import os
import random
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

os.makedirs("data/processed", exist_ok=True)


class SimpleSTS:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def score(self, a, b):
        e1 = self.embed(a)
        e2 = self.embed(b)
        return float(F.cosine_similarity(e1, e2).item())


sts_model = SimpleSTS()


def detect_contradiction(student):
    neg_words = ["not", "never", "incorrect", "wrong"]
    return any(w in student.lower() for w in neg_words)


def assign_gap(correct, student):
    score = sts_model.score(correct, student)

    if score > 0.85:
        return "correct", score
    elif score > 0.5:
        return "incomplete", score
    else:
        if detect_contradiction(student):
            return "misconception", score
        else:
            return "missing_concept", score


def generate_synthetic(correct):
    words = correct.split()

    if len(words) < 5:
        return [correct]

    partial = " ".join(words[:-2])
    wrong = "This is incorrect because " + words[0]

    return [correct, partial, wrong]


def build_race():
    data = load_dataset("race", "all", split="train[:500]")
    rows = []

    for ex in data:
        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        correct = ex["options"][answer_map[ex["answer"]]]

        for student in generate_synthetic(correct):
            gap, score = assign_gap(correct, student)

            rows.append({
                "question": ex["question"],
                "correct_answer": correct,
                "student_answer": student,
                "gap_type": gap,
                "gap_score": score,
                "socratic_question": "Why do you think that is the case?"
            })

    return rows


def build_sciq():
    data = load_dataset("sciq", split="train[:500]")
    rows = []

    for ex in data:
        correct = ex["correct_answer"]

        for student in generate_synthetic(correct):
            gap, score = assign_gap(correct, student)

            rows.append({
                "question": ex["question"],
                "correct_answer": correct,
                "student_answer": student,
                "gap_type": gap,
                "gap_score": score,
                "socratic_question": "Can you explain your reasoning?"
            })

    return rows


def main():
    data = build_race() + build_sciq()
    random.shuffle(data)

    df = pd.DataFrame(data)

    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    val = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(val.index)

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train),
        "validation": Dataset.from_pandas(val),
        "test": Dataset.from_pandas(test)
    })

    dataset.save_to_disk("data/processed/")
    df.to_csv("data/socratic_dataset.csv", index=False)


if __name__ == "__main__":
    main()
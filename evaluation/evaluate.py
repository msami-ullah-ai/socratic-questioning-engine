import json
from datasets import load_from_disk
from bert_score import score
from rouge_score import rouge_scorer

dataset = load_from_disk("data/processed/")["test"]

def evaluate():
    preds = [ex["socratic_question"] for ex in dataset]
    refs = preds  # placeholder

    P, R, F1 = score(preds, refs, lang="en")

    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(p, r)["rougeL"].fmeasure for p, r in zip(preds, refs)]

    report = {
        "bert_score": float(F1.mean()),
        "rougeL": sum(rouge_scores)/len(rouge_scores)
    }

    with open("evaluation/results/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    evaluate()
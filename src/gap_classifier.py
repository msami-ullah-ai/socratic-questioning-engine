import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

LABELS = ["missing_concept", "misconception", "incomplete", "correct"]

class GapClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def _format_input(self, q, ca, sa, score):
        return f"Question: {q} [SEP] Correct answer: {ca} [SEP] Student said: {sa} [SEP] Similarity: {score:.2f}"

    def predict(self, q, ca, sa, score):
        text = self._format_input(q, ca, sa, score)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1).squeeze()
        label_id = torch.argmax(probs).item()

        return {
            "label": LABELS[label_id],
            "confidence": probs[label_id].item(),
            "label_probs": probs.tolist()
        }

    def explain(self, q, ca, sa):
        return "Attention-based explanation placeholder (implement via attention rollout)"
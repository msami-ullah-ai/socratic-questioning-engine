from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class SimilarityScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def _embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def score(self, correct_answer, student_answer):
        emb1 = self._embed(correct_answer)
        emb2 = self._embed(student_answer)
        return float(F.cosine_similarity(emb1, emb2).item())

    def batch_score(self, pairs):
        scores = []
        for a, b in pairs:
            scores.append(self.score(a, b))
        return scores

    def score_with_concepts(self, correct_answer, student_answer):
        parts = correct_answer.split(".")
        present, missing = [], []

        for p in parts:
            if not p.strip():
                continue
            s = self.score(p, student_answer)
            if s > 0.6:
                present.append(p)
            else:
                missing.append(p)

        return {
            "overall_score": self.score(correct_answer, student_answer),
            "present_concepts": present,
            "missing_concepts": missing
        }
from src.similarity_scorer import SimilarityScorer
from src.gap_classifier import GapClassifier
from src.intent_selector import IntentSelector
from src.question_generator import SocraticGenerator


class SocraticPipeline:
    def __init__(self):
        self.sim = SimilarityScorer()
        self.classifier = GapClassifier("./models/gap_classifier")
        self.intent = IntentSelector()

        self.generator = SocraticGenerator(
            "microsoft/phi-2"
        )

    def run_turn(self, q, ca, sa, history=[]):
        score = self.sim.score(ca, sa)

        pred = self.classifier.predict(q, ca, sa, score)

        intent = self.intent.select_intent(pred["label"], history, sa)

        question = self.generator.generate(
            q, ca, sa,
            pred["label"],
            intent["intent"],
            intent["strategy"],
            history
        )

        return {
            "gap_score": score,
            "gap_type": pred["label"],
            "intent": intent["intent"],
            "strategy": intent["strategy"],
            "socratic_question": question,
            "turn_number": len(history) + 1,
            "is_complete": pred["label"] == "correct"
        }
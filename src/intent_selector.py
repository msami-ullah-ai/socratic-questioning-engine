class IntentSelector:
    def select_intent(self, gap_type, dialogue_history, student_answer):
        turns = len(dialogue_history)

        if gap_type == "missing_concept":
            if turns <= 2:
                return {"intent": "probe_for_missing_idea",
                        "strategy": "Ask about missing concept",
                        "constraints": []}
            else:
                return {"intent": "provide_conceptual_scaffold",
                        "strategy": "Hint toward missing idea",
                        "constraints": []}

        if gap_type == "misconception":
            if turns <= 2:
                return {"intent": "surface_contradiction",
                        "strategy": "Ask contradiction question",
                        "constraints": []}
            else:
                return {"intent": "guided_correction",
                        "strategy": "Step-by-step correction",
                        "constraints": []}

        if gap_type == "incomplete":
            return {"intent": "deepen_understanding",
                    "strategy": "Ask for elaboration",
                    "constraints": []}

        return {"intent": "affirm_and_extend",
                "strategy": "Ask harder question",
                "constraints": []}
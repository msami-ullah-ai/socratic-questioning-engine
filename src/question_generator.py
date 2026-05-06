import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SocraticGenerator:
    def __init__(self, base_model, adapter_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        self.model.to(self.device)

    def _build_prompt(self, q, ca, sa, gap, intent, strategy, history):
        history_text = "\n".join(history) if history else "None"

        return f"""[INST]
You are an expert Socratic tutor.

CRITICAL RULES:
1. NEVER reveal or state the correct answer
2. Ask exactly ONE question
3. Be guiding, not telling

Question: {q}
Correct answer (DO NOT REVEAL): {ca}
Student answer: {sa}
Gap type: {gap}
Intent: {intent}
Strategy: {strategy}
Previous dialogue: {history_text}

Generate one Socratic question:
[/INST]
"""

    def generate(self, q, ca, sa, gap, intent, strategy, history, max_new_tokens=100):
        prompt = self._build_prompt(q, ca, sa, gap, intent, strategy, history)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text.split("[/INST]")[-1].strip()

    def generate_with_beam_search(self, q, ca, sa, gap, intent, strategy, history):
        prompt = self._build_prompt(q, ca, sa, gap, intent, strategy, history)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=3,
            max_new_tokens=100
        )

        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def check_answer_leak(self, generated_question, correct_answer):
        return correct_answer.lower() in generated_question.lower()
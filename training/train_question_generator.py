from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
import torch

dataset = load_from_disk("data/processed/")["train"]

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def format_example(ex):
    return f"""[INST]
You are a Socratic tutor.

Question: {ex['question']}
Correct answer: {ex['correct_answer']}
Student answer: {ex['student_answer']}
Gap type: {ex['gap_type']}

Generate one Socratic question:
[/INST]
{ex['socratic_question']}"""

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_example,
    max_seq_length=512,
    args=dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        output_dir="./models/socratic_generator"
    )
)

trainer.train()
model.save_pretrained("./models/socratic_generator")
from src.question_generator import SocraticGenerator

def test_generator():
    gen = SocraticGenerator(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "./models/socratic_generator"
    )

    q = gen.generate(
        "What is gravity?",
        "Force of attraction",
        "It pushes things",
        "misconception",
        "surface_contradiction",
        "Ask contradiction",
        []
    )

    assert isinstance(q, str)
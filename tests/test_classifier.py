from src.gap_classifier import GapClassifier

def test_classifier():
    model = GapClassifier("./models/gap_classifier")

    out = model.predict(
        "What is gravity?",
        "Force that attracts objects",
        "It pushes objects away",
        0.2
    )

    assert "label" in out
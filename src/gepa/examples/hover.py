# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# HoVer: Many-hop fact extraction and claim verification
# Dataset: hover-nlp/hover on HuggingFace
# Labels: "SUPPORTED" / "NOT_SUPPORTED"


_LABEL_MAP = {0: "NOT_SUPPORTED", 1: "SUPPORTED"}


def hover_evaluator(data, response: str):
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult

    gold_label = data["answer"]
    response_upper = response.upper()

    if gold_label == "SUPPORTED":
        correct = "SUPPORTED" in response_upper and "NOT_SUPPORTED" not in response_upper
    else:
        correct = "NOT_SUPPORTED" in response_upper or ("NOT" in response_upper and "SUPPORTED" in response_upper)

    score = 1.0 if correct else 0.0
    if correct:
        feedback = f"Correct. The claim is '{gold_label}'."
    else:
        feedback = f"Wrong. Expected '{gold_label}'. Got: '{response[:200]}'"

    return EvaluationResult(score=score, feedback=feedback)


def init_dataset(train_size: int = 200, val_size: int = 100):
    import random

    from datasets import load_dataset

    raw = load_dataset("hover-nlp/hover")

    def _fmt(x):
        label = _LABEL_MAP.get(x["label"], "NOT_SUPPORTED")
        return {
            "input": x["claim"],
            "answer": label,
            "additional_context": {},
        }

    train_split = [_fmt(x) for x in raw["train"]]
    val_split = [_fmt(x) for x in raw["validation"]]

    rng = random.Random(0)
    rng.shuffle(train_split)
    rng.shuffle(val_split)

    trainset = train_split[:train_size]
    valset = val_split[:val_size]
    testset = val_split[val_size : val_size * 2]

    return trainset, valset, testset

# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# HotpotQA multi-hop setup. Each DataInst carries the 10 distractor paragraphs
# plus the gold supporting titles; retrieval is performed inside
# MultiHopHotpotAdapter so GEPA can optimize 4 prompts end-to-end.


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hotpotqa_evaluator(data, response: str):
    """Kept for backwards compat with single-prompt runs; not used by the
    multi-hop adapter (which scores internally).
    """
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult

    answer = data["answer"]
    f1 = _f1_score(response, answer)
    exact = 1.0 if answer.lower().strip() in response.lower() else 0.0
    score = max(f1, exact)

    if score > 0:
        feedback = f"Correct. F1={f1:.2f}. Expected: '{answer}'"
    else:
        feedback = f"Wrong. Expected: '{answer}'. Got: '{response[:200]}'"

    return EvaluationResult(score=score, feedback=feedback)


def init_dataset(train_size: int = 150, val_size: int = 300):
    """Load HotpotQA distractor and emit multi-hop records.

    Returns three lists of dicts with shape:
        {
          "question": str,
          "answer": str,
          "paragraphs": [{"title": str, "text": str}, ...],   # 10 paragraphs
          "supporting_titles": [str, ...],                     # gold-supporting paragraph titles
          # back-compat fields for any code still using the single-prompt path:
          "input": str,
          "additional_context": {},
        }
    """
    import random

    from datasets import load_dataset

    raw = load_dataset("hotpot_qa", "distractor")

    def _fmt(x):
        ctx = x["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        paragraphs = [
            {"title": t, "text": " ".join(s)}
            for t, s in zip(titles, sentences, strict=True)
        ]
        supporting_titles = list(dict.fromkeys(x["supporting_facts"]["title"]))
        return {
            "question": x["question"],
            "answer": x["answer"],
            "paragraphs": paragraphs,
            "supporting_titles": supporting_titles,
            # back-compat: some callers expect `input` / `additional_context`
            "input": x["question"],
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

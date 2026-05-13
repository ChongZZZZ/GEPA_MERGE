# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# IFBench: Generalizing Verifiable Instruction Following
# Dataset: allenai/IFBench on HuggingFace
#
# Each example has a list of verifiable constraints. Score = fraction satisfied.


def _check_constraints(response: str, constraints: list[dict]) -> tuple[float, list[str]]:
    """
    Check each constraint against the response.
    Returns (fraction_satisfied, list_of_feedback_strings).

    Supported constraint types (subset used in IFBench):
      - contains_keyword: response must include a keyword
      - not_contains_keyword: response must not include a keyword
      - word_count_less_than: response word count < threshold
      - word_count_greater_than: response word count > threshold
      - starts_with: response starts with a string
      - ends_with: response ends with a string
      - language: rough check — just flag as unverifiable locally
    """
    results: list[str] = []
    satisfied = 0

    for c in constraints:
        ctype = c.get("type", "")
        value = c.get("value", "")
        lower_resp = response.lower()

        if ctype == "contains_keyword":
            ok = str(value).lower() in lower_resp
        elif ctype == "not_contains_keyword":
            ok = str(value).lower() not in lower_resp
        elif ctype == "word_count_less_than":
            ok = len(response.split()) < int(value)
        elif ctype == "word_count_greater_than":
            ok = len(response.split()) > int(value)
        elif ctype == "starts_with":
            ok = response.strip().lower().startswith(str(value).lower())
        elif ctype == "ends_with":
            ok = response.strip().lower().endswith(str(value).lower())
        else:
            # Unknown constraint type — skip, don't penalize
            continue

        if ok:
            satisfied += 1
            results.append(f"[PASS] {ctype}={value}")
        else:
            results.append(f"[FAIL] {ctype}={value}")

    total = len(results)
    score = satisfied / total if total > 0 else 0.0
    return score, results


def ifbench_evaluator(data, response: str):
    from gepa.adapters.default_adapter.default_adapter import EvaluationResult

    constraints = data["additional_context"].get("constraints", [])
    score, details = _check_constraints(response, constraints)
    feedback_lines = "\n".join(details)

    if score == 1.0:
        feedback = f"All constraints satisfied.\n{feedback_lines}"
    else:
        feedback = f"Score {score:.2f}: some constraints failed.\n{feedback_lines}"

    return EvaluationResult(score=score, feedback=feedback)


def init_dataset(train_size: int = 200, val_size: int = 100):
    import random

    from datasets import load_dataset

    raw = load_dataset("allenai/IFBench")

    def _fmt(x):
        constraints = x.get("constraints", [])
        if isinstance(constraints, str):
            import json

            try:
                constraints = json.loads(constraints)
            except Exception:
                constraints = []
        return {
            "input": x["instruction"],
            "answer": "",  # IFBench has no single gold answer; score via constraints
            "additional_context": {"constraints": constraints},
        }

    splits = list(raw.keys())
    train_split_name = "train" if "train" in splits else splits[0]
    val_split_name = "validation" if "validation" in splits else splits[-1]

    train_split = [_fmt(x) for x in raw[train_split_name]]
    val_split = [_fmt(x) for x in raw[val_split_name]]

    rng = random.Random(0)
    rng.shuffle(train_split)
    rng.shuffle(val_split)

    trainset = train_split[:train_size]
    valset = val_split[:val_size]
    testset = val_split[val_size : val_size * 2]

    return trainset, valset, testset

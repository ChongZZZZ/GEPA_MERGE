"""2WikiMultiHopQA 4-predictor multi-hop program.

Mirrors MusiqueMultiHop (summarize1 → create_query_hop2 → summarize2 →
final_answer) so the merge-ablation surface is identical across the
in-context multi-hop QA tasks. Per-predictor feedback surfaces
2Wiki's distinctive evidence-triple chain.
"""

from __future__ import annotations

import dspy

from .. import dspy_program


def _format_paragraphs(paragraphs) -> str:
    parts = []
    for p in paragraphs:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        idx = p.get("idx", len(parts))
        parts.append(f"[{idx}] {title} | {text}")
    return "\n".join(parts)


class TwoWikiMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize1 = dspy.ChainOfThought("question, paragraphs -> summary")
        self.create_query_hop2 = dspy.ChainOfThought("question, summary_1 -> query")
        self.summarize2 = dspy.ChainOfThought("question, context, paragraphs -> summary")
        self.final_answer = dspy.ChainOfThought("question, summary_1, summary_2 -> answer")

    def forward(self, question, paragraphs):
        paragraph_block = _format_paragraphs(paragraphs)

        summary_1 = self.summarize1(
            question=question, paragraphs=paragraph_block
        ).summary

        hop2_query = self.create_query_hop2(
            question=question, summary_1=summary_1
        ).query

        summary_2 = self.summarize2(
            question=question,
            context=f"Hop-1 summary:\n{summary_1}\n\nHop-2 focus query:\n{hop2_query}",
            paragraphs=paragraph_block,
        ).summary

        answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer
        return dspy.Prediction(answer=answer, summary_1=summary_1, summary_2=summary_2)


# ---------------------------------------------------------------------------
# Metric + per-predictor feedback
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    import re
    import string

    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def answer_match_fn(prediction: str, answers, frac: float = 1.0) -> bool:
    if isinstance(answers, str):
        answers = [answers]
    pred_norm = _normalize(prediction or "")
    gold_norms = [_normalize(a) for a in answers if a]
    if any(pred_norm == g for g in gold_norms):
        return True
    if frac >= 1.0:
        return False

    pred_toks = set(pred_norm.split())
    for g in gold_norms:
        gold_toks = set(g.split())
        if not gold_toks or not pred_toks:
            continue
        common = pred_toks & gold_toks
        if not common:
            continue
        precision = len(common) / len(pred_toks)
        recall = len(common) / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 >= frac:
            return True
    return False


def _evidence_chain(evidences) -> str:
    """Render 2Wiki's (subject, relation, object) triples as a readable chain."""
    if not evidences:
        return "  (no evidence triples provided)"
    lines = []
    for triple in evidences:
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            s, r, o = triple[0], triple[1], triple[2]
            lines.append(f"  - ({s})  --[{r}]-->  ({o})")
        else:
            lines.append(f"  - {triple}")
    return "\n".join(lines)


def _supporting_titles(paragraphs) -> list[str]:
    return sorted({p.get("title", "") for p in paragraphs if p.get("is_supporting")})


def twowiki_metric_with_feedback(example, pred, trace=None, frac=1.0):
    gold = example.answer
    correct = answer_match_fn(pred.answer, [gold], frac=frac)
    chain = _evidence_chain(getattr(example, "evidences", []))
    titles = _supporting_titles(getattr(example, "paragraphs", []))
    qtype = getattr(example, "qtype", "")

    if correct:
        feedback = (
            f"The provided answer '{pred.answer}' is correct.\n"
            f"Gold answer: '{gold}'  (question type: {qtype}).\n"
            f"Supporting paragraphs (titles): {titles}\n"
            f"Evidence chain:\n{chain}"
        )
    else:
        feedback = (
            f"The provided answer '{pred.answer}' is incorrect. "
            f"The correct answer is '{gold}'  (question type: {qtype}).\n"
            f"Supporting paragraphs (titles): {titles}\n"
            f"Evidence chain you should have followed:\n{chain}"
        )
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)


def _final_score(module_inputs, module_outputs) -> float:
    return 1.0 if answer_match_fn(module_outputs.get("answer", ""), [module_inputs.answer]) else 0.0


def provide_feedback_to_answer_module(
    predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace
):
    res = twowiki_metric_with_feedback(module_inputs, dspy.Prediction(**module_outputs))
    return {"feedback_score": res.score, "feedback_text": res.feedback}


def _summary_feedback(module_inputs, module_outputs, hop_label: str):
    score = _final_score(module_inputs, module_outputs)
    titles = _supporting_titles(module_inputs.paragraphs)
    chain = _evidence_chain(getattr(module_inputs, "evidences", []))
    qtype = getattr(module_inputs, "qtype", "")
    text = (
        f"You are the {hop_label} summarization module in a multi-hop QA system. "
        f"Your output feeds the answer module which has no other context.\n\n"
        f"Question ({qtype}): \"{module_inputs.question}\"\n"
        f"Gold answer: \"{module_inputs.answer}\"\n"
        f"Final program answer: \"{module_outputs.get('answer', '')}\"\n"
        f"Supporting paragraphs (titles): {titles}\n"
        f"Gold evidence chain:\n{chain}\n\n"
        f"Tip: surface the bridge entities and relations named in the chain "
        f"above; missing any of them blocks the next hop."
    )
    return {"feedback_score": score, "feedback_text": text}


def provide_feedback_to_summarize1_module(
    predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace
):
    return _summary_feedback(module_inputs, module_outputs, "first-hop")


def provide_feedback_to_summarize2_module(
    predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace
):
    return _summary_feedback(module_inputs, module_outputs, "second-hop")


def provide_feedback_to_query_module(
    predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace
):
    score = _final_score(module_inputs, module_outputs)
    chain = _evidence_chain(getattr(module_inputs, "evidences", []))
    qtype = getattr(module_inputs, "qtype", "")
    text = (
        f"You are generating the focus query for the **second hop** of a "
        f"multi-hop QA system.\n\n"
        f"Question ({qtype}): \"{module_inputs.question}\"\n"
        f"Gold answer: \"{module_inputs.answer}\"\n"
        f"Hop-1 summary the next stage will see:\n"
        f"{module_outputs.get('summary_1', '')}\n\n"
        f"Gold evidence chain (target):\n{chain}\n\n"
        f"A good query reframes what is still missing after the hop-1 summary "
        f"into a focused lookup. Avoid restating the original question verbatim."
    )
    return {"feedback_score": score, "feedback_text": text}


feedback_fn_map = {
    "summarize1.predict": provide_feedback_to_summarize1_module,
    "create_query_hop2.predict": provide_feedback_to_query_module,
    "summarize2.predict": provide_feedback_to_summarize2_module,
    "final_answer.predict": provide_feedback_to_answer_module,
}

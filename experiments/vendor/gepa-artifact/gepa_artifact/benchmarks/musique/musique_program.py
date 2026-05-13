"""MuSiQue 4-predictor multi-hop program.

Mirrors the structure of HotpotMultiHop (summarize1 → create_query_hop2 →
summarize2 → final_answer) so the merge ablation surface is comparable
across the two multi-hop tasks. The key difference: MuSiQue ships
candidate paragraphs with each example (no external retriever), so the
"hop 1" / "hop 2" rounds both look at the same 20-paragraph pool but
condition on different prior context.
"""

from __future__ import annotations

import dspy

from .. import dspy_program


def _format_paragraphs(paragraphs) -> str:
    """Render the per-example paragraph list as a single string suitable for
    feeding into a CoT predictor."""
    parts = []
    for p in paragraphs:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        idx = p.get("idx", len(parts))
        parts.append(f"[{idx}] {title} | {text}")
    return "\n".join(parts)


class MusiqueMultiHop(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
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

        # Hop-2 query helps reframe what the model still needs to find.
        # On MuSiQue we don't actually run a retriever; the second-stage
        # summarization re-reads the same paragraphs with the focused query
        # injected via `context`.
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

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def answer_match_fn(prediction: str, answers, frac: float = 1.0) -> bool:
    """Robust answer matcher: tries EM after normalisation against the gold
    answer and any aliases. Falls back to F1 token overlap if `frac < 1.0`.
    """
    if isinstance(answers, str):
        answers = [answers]
    pred_norm = _normalize(prediction or "")
    gold_norms = [_normalize(a) for a in answers if a]
    if any(pred_norm == g for g in gold_norms):
        return True
    if frac >= 1.0:
        return any(pred_norm == g for g in gold_norms)

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


def _supporting_titles(paragraphs) -> list[str]:
    return sorted({p.get("title", "") for p in paragraphs if p.get("is_supporting")})


def _decomposition_chain(decomposition) -> str:
    """Render question_decomposition as a plain readable chain."""
    lines = []
    for step in decomposition or []:
        q = step.get("question", "")
        a = step.get("answer", "")
        lines.append(f"  - {q}  →  {a}")
    return "\n".join(lines) if lines else "  (no decomposition provided)"


def musique_metric_with_feedback(example, pred, trace=None, frac=1.0):
    gold = example.answer
    aliases = list(getattr(example, "answer_aliases", []) or [])
    gold_set = [gold, *aliases]
    correct = answer_match_fn(pred.answer, gold_set, frac=frac)
    chain = _decomposition_chain(getattr(example, "question_decomposition", []))
    titles = _supporting_titles(getattr(example, "paragraphs", []))

    if correct:
        feedback = (
            f"The provided answer '{pred.answer}' is correct.\n"
            f"Gold answer: '{gold}'.\n"
            f"Supporting paragraphs (titles): {titles}\n"
            f"Reasoning chain:\n{chain}"
        )
    else:
        feedback = (
            f"The provided answer '{pred.answer}' is incorrect. "
            f"The correct answer is '{gold}'"
            f"{(' (aliases: ' + ', '.join(aliases) + ')') if aliases else ''}.\n"
            f"Supporting paragraphs (titles): {titles}\n"
            f"Reasoning chain you should have followed:\n{chain}"
        )
    return dspy.Prediction(score=1.0 if correct else 0.0, feedback=feedback)


def _final_score(module_inputs, module_outputs) -> float:
    aliases = list(getattr(module_inputs, "answer_aliases", []) or [])
    gold_set = [module_inputs.answer, *aliases]
    return 1.0 if answer_match_fn(module_outputs.get("answer", ""), gold_set) else 0.0


def provide_feedback_to_answer_module(
    predictor_output, predictor_inputs, module_inputs, module_outputs, captured_trace
):
    res = musique_metric_with_feedback(module_inputs, dspy.Prediction(**module_outputs))
    return {"feedback_score": res.score, "feedback_text": res.feedback}


def _summary_feedback(module_inputs, module_outputs, hop_label: str):
    score = _final_score(module_inputs, module_outputs)
    titles = _supporting_titles(module_inputs.paragraphs)
    chain = _decomposition_chain(getattr(module_inputs, "question_decomposition", []))
    text = (
        f"You are the {hop_label} summarization module in a multi-hop QA system. "
        f"Your output feeds the answer module which has no other context.\n\n"
        f"Question: \"{module_inputs.question}\"\n"
        f"Gold answer: \"{module_inputs.answer}\"\n"
        f"Final program answer: \"{module_outputs.get('answer', '')}\"\n"
        f"Supporting paragraphs (titles): {titles}\n"
        f"Gold reasoning chain:\n{chain}\n\n"
        f"Tip: ensure your summary surfaces the bridge entities and facts "
        f"named in the chain above; missing any of them blocks the next hop."
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
    chain = _decomposition_chain(getattr(module_inputs, "question_decomposition", []))
    text = (
        f"You are generating the focus query for the **second hop** of a "
        f"multi-hop QA system.\n\n"
        f"Question: \"{module_inputs.question}\"\n"
        f"Gold answer: \"{module_inputs.answer}\"\n"
        f"Hop-1 summary the next stage will see:\n"
        f"{module_outputs.get('summary_1', '')}\n\n"
        f"Gold reasoning chain (target):\n{chain}\n\n"
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

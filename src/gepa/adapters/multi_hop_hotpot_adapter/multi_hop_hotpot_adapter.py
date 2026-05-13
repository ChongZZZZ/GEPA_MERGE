# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# Multi-hop HotpotQA adapter: K=4 module program mirroring the GEPA paper's
# HoVerMultiHop-derived setup (Section L.1, HotpotQA GPT-4.1-Mini).
#
# Modules:
#   summarize1       : (question, passages)              -> summary_1
#   create_query_hop2: (question, summary_1)             -> query
#   summarize2       : (question, context, passages)     -> summary_2
#   final_answer     : (question, summary_1, summary_2)  -> answer
#
# Retrieval is lexical (token-overlap BM25-lite) over the 10 distractor
# paragraphs provided by the HotpotQA `distractor` config; no external
# index is required.

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict, cast

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


MODULE_NAMES = ("summarize1", "create_query_hop2", "summarize2", "final_answer")


class Paragraph(TypedDict):
    title: str
    text: str


class MultiHopHotpotDataInst(TypedDict):
    question: str
    answer: str
    paragraphs: list[Paragraph]
    supporting_titles: list[str]


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class MultiHopTrajectory(TypedDict):
    data: MultiHopHotpotDataInst
    hop1_titles: list[str]
    hop2_titles: list[str]
    module_io: dict[str, dict[str, str]]
    final_response: str
    feedback: str
    score: float


class MultiHopRolloutOutput(TypedDict):
    final_response: str
    summary_1: str
    summary_2: str
    hop2_query: str
    hop1_titles: list[str]
    hop2_titles: list[str]


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _bm25_lite(query: str, paragraphs: list[Paragraph], k: int) -> list[int]:
    """Return indices of top-k paragraphs by BM25-ish lexical score.

    Uses standard BM25 (k1=1.5, b=0.75) over simple whitespace/alphanum tokens,
    scoring each paragraph as `title + "\n" + text` against the query. Pure
    stdlib; no sklearn / rank_bm25 dependency.
    """
    if not paragraphs:
        return []
    docs = [_tokenize(p["title"] + "\n" + p["text"]) for p in paragraphs]
    avgdl = sum(len(d) for d in docs) / max(1, len(docs))
    # document frequency
    df: dict[str, int] = {}
    for d in docs:
        for tok in set(d):
            df[tok] = df.get(tok, 0) + 1
    N = len(docs)
    k1, b = 1.5, 0.75
    q_tokens = _tokenize(query)
    scores: list[float] = []
    for d in docs:
        if not d:
            scores.append(0.0)
            continue
        dl = len(d)
        tf: dict[str, int] = {}
        for tok in d:
            tf[tok] = tf.get(tok, 0) + 1
        s = 0.0
        for tok in q_tokens:
            if tok not in tf:
                continue
            idf = math.log(1 + (N - df[tok] + 0.5) / (df[tok] + 0.5))
            num = tf[tok] * (k1 + 1)
            den = tf[tok] + k1 * (1 - b + b * dl / max(1.0, avgdl))
            s += idf * num / den
        scores.append(s)
    ranked = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
    return ranked[: max(0, k)]


def _format_passages(paragraphs: list[Paragraph], indices: list[int]) -> str:
    out = []
    for i in indices:
        p = paragraphs[i]
        out.append(f"[Title: {p['title']}]\n{p['text']}")
    return "\n\n".join(out)


class MultiHopHotpotAdapter(
    GEPAAdapter[MultiHopHotpotDataInst, MultiHopTrajectory, MultiHopRolloutOutput]
):
    """GEPAAdapter for a 4-module multi-hop HotpotQA program."""

    def __init__(
        self,
        model: str | ChatCompletionCallable,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
        hop_k: int = 3,
    ):
        if isinstance(model, str):
            import litellm  # local import

            self.litellm = litellm
        self.model = model
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        self.hop_k = hop_k

    # -------- LLM plumbing --------

    def _batch_chat(self, messages_list: list[list[ChatMessage]]) -> list[str]:
        """Run a list of chat requests, return one assistant string per request."""
        if not messages_list:
            return []
        if isinstance(self.model, str):
            responses = self.litellm.batch_completion(
                model=self.model,
                messages=messages_list,
                max_workers=self.max_litellm_workers,
                **self.litellm_batch_completion_kwargs,
            )
            out: list[str] = []
            for resp in responses:
                try:
                    content = resp.choices[0].message.content
                    out.append((content or "").strip())
                except Exception as e:
                    # Soft-fail this example so a single rate-limit/parse error
                    # doesn't crash the whole minibatch.
                    out.append(f"[ERROR: {type(e).__name__}: {str(e)[:200]}]")
            return out
        else:
            return [self.model(m) for m in messages_list]

    # -------- Module prompt construction --------

    @staticmethod
    def _user_msg_summarize1(question: str, hop1_passages: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Passages:\n{hop1_passages}\n\n"
            "Produce the field `summary`."
        )

    @staticmethod
    def _user_msg_query_hop2(question: str, summary_1: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Summary 1: {summary_1}\n\n"
            "Produce the field `query` (a short search query for the missing information)."
        )

    @staticmethod
    def _user_msg_summarize2(question: str, summary_1: str, hop2_passages: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Context (Summary 1): {summary_1}\n\n"
            f"Passages:\n{hop2_passages}\n\n"
            "Produce the field `summary`."
        )

    @staticmethod
    def _user_msg_final(question: str, summary_1: str, summary_2: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Summary 1: {summary_1}\n\n"
            f"Summary 2: {summary_2}\n\n"
            "Produce the field `answer`. Keep it concise (a span or short phrase)."
        )

    # -------- Core evaluation --------

    def evaluate(
        self,
        batch: list[MultiHopHotpotDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MultiHopTrajectory, MultiHopRolloutOutput]:
        for m in MODULE_NAMES:
            if m not in candidate:
                raise ValueError(
                    f"Candidate is missing required module '{m}'. Got keys: {list(candidate.keys())}"
                )

        n = len(batch)

        # --- Hop 1 retrieval (pure local) ---
        hop1_indices: list[list[int]] = [
            _bm25_lite(ex["question"], ex["paragraphs"], self.hop_k) for ex in batch
        ]
        hop1_strs = [_format_passages(batch[i]["paragraphs"], hop1_indices[i]) for i in range(n)]

        # --- Module 1: summarize1 ---
        sys1 = candidate["summarize1"]
        msgs1: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": sys1},
                {"role": "user", "content": self._user_msg_summarize1(ex["question"], hop1_strs[i])},
            ]
            for i, ex in enumerate(batch)
        ]
        summary_1s = self._batch_chat(msgs1)

        # --- Module 2: create_query_hop2 ---
        sys2 = candidate["create_query_hop2"]
        msgs2: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": sys2},
                {"role": "user", "content": self._user_msg_query_hop2(ex["question"], summary_1s[i])},
            ]
            for i, ex in enumerate(batch)
        ]
        hop2_queries = self._batch_chat(msgs2)

        # --- Hop 2 retrieval: rank remaining paragraphs by the generated query ---
        hop2_indices: list[list[int]] = []
        for i, ex in enumerate(batch):
            used = set(hop1_indices[i])
            remaining = [(idx, p) for idx, p in enumerate(ex["paragraphs"]) if idx not in used]
            if not remaining:
                hop2_indices.append([])
                continue
            rem_paras = [p for _, p in remaining]
            rem_idx_map = [idx for idx, _ in remaining]
            local_top = _bm25_lite(hop2_queries[i], rem_paras, self.hop_k)
            hop2_indices.append([rem_idx_map[j] for j in local_top])
        hop2_strs = [_format_passages(batch[i]["paragraphs"], hop2_indices[i]) for i in range(n)]

        # --- Module 3: summarize2 ---
        sys3 = candidate["summarize2"]
        msgs3: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": sys3},
                {
                    "role": "user",
                    "content": self._user_msg_summarize2(ex["question"], summary_1s[i], hop2_strs[i]),
                },
            ]
            for i, ex in enumerate(batch)
        ]
        summary_2s = self._batch_chat(msgs3)

        # --- Module 4: final_answer ---
        sys4 = candidate["final_answer"]
        msgs4: list[list[ChatMessage]] = [
            [
                {"role": "system", "content": sys4},
                {
                    "role": "user",
                    "content": self._user_msg_final(ex["question"], summary_1s[i], summary_2s[i]),
                },
            ]
            for i, ex in enumerate(batch)
        ]
        final_responses = self._batch_chat(msgs4)

        # --- Scoring + trajectories ---
        outputs: list[MultiHopRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[MultiHopTrajectory] | None = [] if capture_traces else None

        for i, ex in enumerate(batch):
            resp = final_responses[i]
            answer = ex["answer"]
            f1 = _f1_score(resp, answer)
            exact = 1.0 if answer.lower().strip() in resp.lower() else 0.0
            score = max(f1, exact)

            hop1_titles = [ex["paragraphs"][j]["title"] for j in hop1_indices[i]]
            hop2_titles = [ex["paragraphs"][j]["title"] for j in hop2_indices[i]]
            retrieved_titles = set(hop1_titles) | set(hop2_titles)
            gold_titles = set(ex.get("supporting_titles", []))
            gold_hit = len(retrieved_titles & gold_titles)
            gold_needed = max(1, len(gold_titles))

            output: MultiHopRolloutOutput = {
                "final_response": resp,
                "summary_1": summary_1s[i],
                "summary_2": summary_2s[i],
                "hop2_query": hop2_queries[i],
                "hop1_titles": hop1_titles,
                "hop2_titles": hop2_titles,
            }
            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                gold_status = (
                    f"Retrieved {gold_hit}/{gold_needed} gold supporting titles "
                    f"({sorted(gold_titles)})."
                )
                if score > 0:
                    final_feedback = (
                        f"Correct. F1={f1:.2f}, exact_contains={int(exact)}. "
                        f"Expected: '{answer}'. {gold_status}"
                    )
                else:
                    final_feedback = (
                        f"Wrong. Expected: '{answer}'. Got: '{resp[:200]}'. {gold_status}"
                    )
                trajectories.append(
                    {
                        "data": ex,
                        "hop1_titles": hop1_titles,
                        "hop2_titles": hop2_titles,
                        "module_io": {
                            "summarize1": {
                                "input_question": ex["question"],
                                "input_passages": hop1_strs[i],
                                "output": summary_1s[i],
                            },
                            "create_query_hop2": {
                                "input_question": ex["question"],
                                "input_summary_1": summary_1s[i],
                                "output": hop2_queries[i],
                            },
                            "summarize2": {
                                "input_question": ex["question"],
                                "input_summary_1": summary_1s[i],
                                "input_passages": hop2_strs[i],
                                "output": summary_2s[i],
                            },
                            "final_answer": {
                                "input_question": ex["question"],
                                "input_summary_1": summary_1s[i],
                                "input_summary_2": summary_2s[i],
                                "output": resp,
                            },
                        },
                        "final_response": resp,
                        "feedback": final_feedback,
                        "score": score,
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None,
        )

    # -------- Reflective dataset --------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MultiHopTrajectory, MultiHopRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories required for reflective dataset."

        ret: dict[str, list[dict[str, Any]]] = {m: [] for m in components_to_update}

        for traj in trajectories:
            data = traj["data"]
            q = data["question"]
            gold = data["answer"]
            gold_titles = sorted(set(data.get("supporting_titles", [])))
            score = traj["score"]
            status = "Correct" if score > 0 else "Wrong"

            for module in components_to_update:
                io = traj["module_io"].get(module)
                if io is None:
                    continue

                if module == "summarize1":
                    inputs = {
                        "question": q,
                        "passages_retrieved_hop1": io["input_passages"],
                        "retrieved_titles_hop1": traj["hop1_titles"],
                        "gold_supporting_titles": gold_titles,
                    }
                    feedback = (
                        f"Final answer was {status}. Expected answer: '{gold}'. "
                        f"Summary_1 will be used (a) as context for generating a hop-2 search query "
                        f"and (b) as evidence for the final answer. It should extract entities and "
                        f"facts relevant to the question from the hop-1 passages, and clearly flag "
                        f"what information is still missing."
                    )
                elif module == "create_query_hop2":
                    inputs = {
                        "question": q,
                        "summary_1": io["input_summary_1"],
                        "retrieved_titles_hop1": traj["hop1_titles"],
                        "retrieved_titles_hop2": traj["hop2_titles"],
                        "gold_supporting_titles": gold_titles,
                    }
                    feedback = (
                        f"Final answer was {status}. Expected answer: '{gold}'. "
                        f"The query produced here is used to retrieve hop-2 passages by lexical "
                        f"BM25 over paragraph titles+text. A good query names the missing entity / "
                        f"relation whose Wikipedia page is needed, using words likely to appear in "
                        f"that page's title and opening sentences."
                    )
                elif module == "summarize2":
                    inputs = {
                        "question": q,
                        "summary_1": io["input_summary_1"],
                        "passages_retrieved_hop2": io["input_passages"],
                        "retrieved_titles_hop2": traj["hop2_titles"],
                        "gold_supporting_titles": gold_titles,
                    }
                    feedback = (
                        f"Final answer was {status}. Expected answer: '{gold}'. "
                        f"Summary_2 is the second evidence piece feeding the final answer. It "
                        f"should focus on information from hop-2 passages that bridges what "
                        f"Summary_1 already knows with what the question still needs."
                    )
                elif module == "final_answer":
                    inputs = {
                        "question": q,
                        "summary_1": io["input_summary_1"],
                        "summary_2": io["input_summary_2"],
                    }
                    feedback = (
                        f"{status}. Expected: '{gold}'. Got: '{io['output'][:200]}'. "
                        f"HotpotQA answers are short: a single entity, number, span, or yes/no. "
                        f"Avoid hedging, explanations, or restating the question."
                    )
                else:
                    continue

                ret[module].append(
                    {
                        "Inputs": inputs,
                        "Generated Outputs": io["output"],
                        "Feedback": feedback,
                    }
                )

        # Every requested module must have at least one record, else
        # ReflectiveMutationProposer will skip the update for this iteration.
        empty_modules = [m for m, items in ret.items() if not items]
        if empty_modules and all(not v for v in ret.values()):
            raise Exception("No valid predictions found for any module.")

        return cast(Mapping[str, Sequence[Mapping[str, Any]]], ret)

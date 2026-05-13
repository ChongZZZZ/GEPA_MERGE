"""2WikiMultiHopQA loader — Ho et al. 2020 (COLING).

Each example carries 10 candidate paragraphs in-context (the standard
"distractor" setting), gold supporting facts as (title, sent_id) pairs, and
an evidence chain of (subject, relation, object) triples. No external
retriever is needed; the program structure mirrors MusiqueMultiHop.

Source: voidful/2WikiMultihopQA mirror (Apache-2.0). The repo ships a
loading script that newer `datasets` versions reject, so we pull the raw
train.json via huggingface_hub and parse it ourselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from ..benchmark import Benchmark


_HF_REPO = "voidful/2WikiMultihopQA"
_TRAIN_FILE = "train.json"  # 167,454 examples; the dev/test JSONs are not required for our 150/300/300 split
_REPO_TYPE = "dataset"


def _load_examples():
    """Return the list of raw example dicts (cached after first download)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=_HF_REPO,
        filename=_TRAIN_FILE,
        repo_type=_REPO_TYPE,
    )
    with open(path) as f:
        return json.load(f)


def _format_paragraphs(context) -> list[dict]:
    """Convert 2Wiki's [title, [sent, sent, ...]] context layout into the
    paragraph-dict format MusiqueMultiHop already understands."""
    paragraphs = []
    for idx, item in enumerate(context or []):
        if not item:
            continue
        title, sentences = item[0], item[1] if len(item) > 1 else []
        text = "".join(sentences) if isinstance(sentences, list) else str(sentences)
        paragraphs.append(
            {
                "idx": idx,
                "title": title,
                "paragraph_text": text,
                "sentences": list(sentences) if isinstance(sentences, list) else [],
            }
        )
    return paragraphs


def _supporting_titles(supporting_facts, paragraphs) -> set[str]:
    """Mark `is_supporting=True` on paragraphs whose title appears in the gold
    supporting_facts list (titles repeat across sent_ids in 2Wiki's format)."""
    if not supporting_facts:
        return set()
    titles = set()
    for fact in supporting_facts:
        if not fact:
            continue
        title = fact[0] if isinstance(fact, (list, tuple)) else fact.get("title")
        if title:
            titles.add(title)
    return titles


class TwoWikiMultiHopQABench(Benchmark):
    """2WikiMultiHopQA — multi-hop QA with in-context paragraphs and a gold
    reasoning chain (evidences). Mirrors MusiqueBench so the same DSPy
    program shape can be reused."""

    def init_dataset(self):
        raw = _load_examples()
        examples = []
        for ex in raw:
            paragraphs = _format_paragraphs(ex.get("context") or [])
            support_titles = _supporting_titles(ex.get("supporting_facts") or [], paragraphs)
            for p in paragraphs:
                p["is_supporting"] = p["title"] in support_titles

            answer = ex.get("answer")
            if not answer:
                continue  # skip unlabeled rows defensively

            examples.append(
                dspy.Example(
                    question=ex["question"],
                    paragraphs=paragraphs,
                    answer=answer,
                    supporting_facts=ex.get("supporting_facts") or [],
                    evidences=ex.get("evidences") or [],
                    qtype=ex.get("type", ""),
                ).with_inputs("question", "paragraphs")
            )
        self.dataset = examples

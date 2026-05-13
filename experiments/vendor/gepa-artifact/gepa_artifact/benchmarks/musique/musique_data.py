from ..benchmark import Benchmark
import dspy
from datasets import load_dataset


class MusiqueBench(Benchmark):
    """MuSiQue (Trivedi et al. 2022) — multi-hop QA with per-example paragraph
    candidates. Uses `dgslibisey/MuSiQue` (mirror of the official answerable
    split). Each example provides 20 paragraphs, of which a few are marked
    `is_supporting=True`; the gold answer chain lives in
    `question_decomposition`.

    Inputs to the DSPy program: `question` and `paragraphs` (list of
    {idx, title, paragraph_text, is_supporting}).
    """

    def init_dataset(self):
        raw = load_dataset("dgslibisey/MuSiQue")
        # Use the train split as the working pool; the Benchmark base class
        # will carve out 150 train / 300 val / 300 test by default.
        examples = []
        for ex in raw["train"]:
            if not ex.get("answerable", True):
                continue
            examples.append(
                dspy.Example(
                    question=ex["question"],
                    paragraphs=ex["paragraphs"],
                    answer=ex["answer"],
                    answer_aliases=ex.get("answer_aliases") or [],
                    question_decomposition=ex.get("question_decomposition") or [],
                ).with_inputs("question", "paragraphs")
            )
        self.dataset = examples

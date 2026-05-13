from ..benchmark import BenchmarkMeta
from .musique_data import MusiqueBench
from .musique_program import (
    MusiqueMultiHop,
    musique_metric_with_feedback,
    feedback_fn_map,
)


def musique_metric(example, pred, trace=None, frac=1.0):
    """Score-only variant matching the (example, pred) signature used by
    DSPy's evaluator. Returns 1.0 / 0.0 like answer_exact_match for the
    other multi-hop benchmarks."""
    res = musique_metric_with_feedback(example, pred, trace=trace, frac=frac)
    return res.score


benchmark = [
    BenchmarkMeta(
        MusiqueBench,
        [MusiqueMultiHop()],
        musique_metric,
        metric_with_feedback=musique_metric_with_feedback,
        feedback_fn_maps=[feedback_fn_map],
    )
]

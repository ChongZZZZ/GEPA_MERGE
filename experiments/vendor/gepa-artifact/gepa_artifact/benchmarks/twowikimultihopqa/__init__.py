from ..benchmark import BenchmarkMeta
from .twowiki_data import TwoWikiMultiHopQABench
from .twowiki_program import (
    TwoWikiMultiHop,
    twowiki_metric_with_feedback,
    feedback_fn_map,
)


def twowiki_metric(example, pred, trace=None, frac=1.0):
    """Score-only variant matching DSPy's evaluator signature; returns 1.0 / 0.0."""
    res = twowiki_metric_with_feedback(example, pred, trace=trace, frac=frac)
    return res.score


benchmark = [
    BenchmarkMeta(
        TwoWikiMultiHopQABench,
        [TwoWikiMultiHop()],
        twowiki_metric,
        metric_with_feedback=twowiki_metric_with_feedback,
        feedback_fn_maps=[feedback_fn_map],
    )
]

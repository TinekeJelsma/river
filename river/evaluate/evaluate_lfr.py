import datetime as dt
import time
import typing

from river import base
from river import metrics
from river import utils
from river import stream
from river.drift import LFR

__all__ = ["evaluate_lfr"]


def evaluate_lfr(
    dataset: base.typing.Stream,
    model,
    lfr: LFR,
    metric: metrics.Metric,
    moment: typing.Union[str, typing.Callable] = None,
    delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
    print_every=0,
    show_time=False,
    show_memory=False,
    **print_kwargs,
) -> metrics.Metric:
    

    # Check that the model and the metric are in accordance
    if not metric.works_with(model):
        raise ValueError(f"{metric.__class__.__name__} metric is not compatible with {model}")

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    if utils.inspect.isclassifier(model) and not metric.requires_labels:
        pred_func = model.predict_proba_one

    preds = {}

    n_total_answers = 0
    if show_time:
        start = time.perf_counter()

    for i, x, y in stream.simulate_qa(dataset, moment, delay, copy=True):

        # Question
        if y is None:
            preds[i] = pred_func(x=x)
            continue

        # Answer
        y_pred = preds.pop(i)
        if y_pred != {} and y_pred is not None:
            metric.update(y_true=y, y_pred=y_pred)
            lfr.update(y_true = y, y_pred = y_pred)
        model.learn_one(x=x, y=y)

        # Update the answer counter
        n_total_answers += 1
        if print_every and not n_total_answers % print_every:
            msg = f"[{n_total_answers:,d}] {metric}"
            if show_time:
                now = time.perf_counter()
                msg += f" – {dt.timedelta(seconds=int(now - start))}"
            if show_memory:
                msg += f" – {model._memory_usage}"
            print(msg, **print_kwargs)

    return metric

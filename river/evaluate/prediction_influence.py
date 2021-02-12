import datetime as dt
from inspect import BoundArguments
import time
import typing
import numpy as np
import pandas as pd
from river import base
from river import metrics
from river import utils
from river import stream
from river.datasets.synth.prediction_influenced_stream import PredictionInfluenceStream
from river.drift import LFR
from scipy.stats import ranksums, kstest
import matplotlib.pyplot as plt
from river import drift
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV



__all__ = ['evaluate_influential']


def evaluate_influential(dataset: base.typing.Stream, model, metric: metrics.Metric,
                         drift_detection: drift.LFR = None,
                         batch_size: int = 1,
                         moment: typing.Union[str, typing.Callable] = None,
                         delay: typing.Union[str, int, dt.timedelta, typing.Callable] = None,
                         print_every=0, max_samples: int = 100, comparison_block: int = 100,
                         prior: int = 1, intervals: int = 6, show_time=False, show_memory=False,
                         **print_kwargs) -> metrics.Metric:
    # Check that the model and the metric are in accordance

    if not metric.works_with(model):
        raise ValueError(f'{metric.__class__.__name__} metric is not compatible with {model}')

    # Determine if predict_one or predict_proba_one should be used in case of a classifier
    pred_func = model.predict_one
    # if utils.inspect.isclassifier(model) and not metric.requires_labels:
    #     print('predict proba')
    #     pred_func = model.predict_proba_one

    preds = {}
    chunk_tracker = 0
    cm = metrics.ConfusionMatrix()
    cache = pd.DataFrame(columns=['timestamp', 'x', 'ypred', 'ytrue'])
    TP, FP, FN, TN = [], [], [], []
    pos_yvalues =  [[] for _ in range(5)]  
    neg_yvalues =  [[] for _ in range(5)]  
    pos_xvalues, neg_xvalues = [], []

    drift_detector_positive = drift.ADWIN()
    drift_detector_negative = drift.ADWIN()
    if batch_size > 1:
        mini_batch_x, mini_batch_y = [], []

    n_total_answers = 0
    if show_time:
        start = time.perf_counter()

    for i, x, y in stream.simulate_qa(dataset, moment, delay, copy=True):
        # Question
        if y is None:
            preds[i] = pred_func(x=x)
            continue
        if n_total_answers > batch_size > 1:
            mini_batch_x.append(x)
            mini_batch_y.append(y)

        # Answer
        y_pred = preds.pop(i)
        if y_pred != {} and y_pred is not None:
            metric.update(y_true=y, y_pred=y_pred)
            if drift_detection is not None:
                # if we use LFR, now update LFR
                drift_detection.update(y_true=y, y_pred=y_pred)
            cm.update(y, y_pred)
            if isinstance(dataset, PredictionInfluenceStream):
                # update weights of stream based on classification
                dataset.receive_feedback(y_true=y, y_pred=y_pred, x_features=x)

            if y == 1:
                key_number = 0
                for key, value in x.items():
                    drift_detector_positive.update(value)  # Data is processed one sample at a time
                    pos_yvalues[key_number].append(float(value))
                    if key_number == 0:
                        # we only need to store the index of positive instance once
                        pos_xvalues.append(n_total_answers)
                    if drift_detector_positive.change_detected:
                        # The drift detector indicates after each sample if there is a drift in the data
                        print(f'Change detected in positively classified at index {i} on feature {key}')
                        drift_detector_positive.reset()
                    key_number += 1

            if y == 0:
                key_number = 0
                for key, value in x.items():
                    drift_detector_negative.update(value)  # Data is processed one sample at a time
                    neg_yvalues[key_number].append(float(value))
                    if key_number == 0:
                        neg_xvalues.append(n_total_answers)
                    if drift_detector_negative.change_detected:
                        # The drift detector indicates after each sample if there is a drift in the data
                        print(f'Change detected  in negatively classified instances at index {i} on feature {key}')
                        drift_detector_negative.reset()
                    key_number += 1

        if len(cache.index) >= comparison_block:
            cache['weight'] = (cache.index + 1) / comparison_block
            subset = cache.loc[(cache['ytrue'] == y) & (cache['ypred'] == y_pred)]
            feature0 = pd.json_normalize(subset['x'])[0].tolist()
            train_x = np.array(feature0)
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_x[:, None])
            score = kde.score_samples(np.array(x.get(0)).reshape(1, -1))
            print(f'score = {score}')
        
        # add instance to cache and remove too old instances
        if len(cache.index) < comparison_block:
            cache.loc[i] = pd.Series({'timestamp': i, 'x': x, 'ypred': y_pred, 'ytrue': y})
        else:
            index = i - comparison_block
            cache = cache.drop(index = index)
            cache.loc[i] = pd.Series({'timestamp': i, 'x': x, 'ypred': y_pred, 'ytrue': y})
            cache = cache.reset_index(drop=True)

        if n_total_answers < batch_size or batch_size == 1:
            # if the model learns online, learn one, or when the index is below the first batch size, learn one
            model.learn_one(x=x, y=y)

        if n_total_answers % batch_size == 0 and n_total_answers >= 2 * batch_size and batch_size != 1:
            # learn many if batch size is bigger than 0, transform the batch in series and df
            mini_batch_y = pd.Series(mini_batch_y)
            mini_batch_x = pd.DataFrame(mini_batch_x)
            model.learn_many(X=mini_batch_x, y=mini_batch_y)
            mini_batch_x, mini_batch_y = [], []        

        # Update the answer counter
        n_total_answers += 1
        if print_every and not n_total_answers % print_every:
            msg = f'[{n_total_answers:,d}] {metric}'
            if show_time:
                now = time.perf_counter()
                msg += f' – {dt.timedelta(seconds=int(now - start))}'
            if show_memory:
                msg += f' – {model._memory_usage}'
            print(msg, **print_kwargs)

        if n_total_answers == max_samples:
            # visualize the feature space in plots
            for i in range(len(x)):
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
                axes[0].plot(pos_xvalues, pos_yvalues[i], label=f'values feature {i}')
                axes[1].plot(neg_xvalues, neg_yvalues[i], label=f'values feature {i}')
                axes[0].title.set_text('Positive instances')
                axes[1].title.set_text('Negative instances')
                plt.legend()

            # if data stream is prediction influence stream, show the weights of the streams
            if isinstance(dataset, PredictionInfluenceStream):
                fig, ax = plt.subplots()
                ax.plot(dataset.weight_tracker)
                # ax.legend(['base negative', 'base positive', 'drift negative', 'drift positive', 'drift negative 2',
                #             'drift positive 2'], loc=0)
            plt.show()

        if n_total_answers >= max_samples:
            print(cm)
            print(cache)
            if isinstance(dataset, PredictionInfluenceStream):
                print(dataset.stream_weight)
            return metric

    return metric

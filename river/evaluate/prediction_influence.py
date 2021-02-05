import datetime as dt
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
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from river import drift

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
    if utils.inspect.isclassifier(model) and not metric.requires_labels:
        pred_func = model.predict_proba_one

    preds = {}
    chunk_tracker = 0
    cm = metrics.ConfusionMatrix()
    TP, FP, FN, TN = [], [], [], []
    cm_values = [TP, FP, FN, TN]
    cm_names = ['TP', 'FP', 'FN', 'TN']
    hist_info = {}

    pos_yvalues =  [[] for _ in range(5)]  
    neg_yvalues =  [[] for _ in range(5)]  
    pvaluePos =  [[] for _ in range(5)]  
    pvalueNeg =  [[] for _ in range(5)]
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
            # put feature values in 4 bins (cm bins)
            if y_pred == 1 and y == 1:
                # true positive
                TP.append(x)
            if y_pred == 1 and y == 0:
                # false positive
                FP.append(x)
            if y_pred == 0 and y == 1:
                # false negative
                FN.append(x)
            if y_pred == 0 and y == 0:
                # true negative
                TN.append(x)

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

        if n_total_answers % comparison_block == 0:
            # fill all bins by looping over keys in feature dict
            feature_names = list(x.keys())
            feature_number = 0
            for feature in feature_names:
                index = 0
                for cm_value in cm_values:
                    feature_values = [value.get(feature) for value in cm_value]
                    values, counts = np.unique(feature_values, return_counts=True)
                    if all(isinstance(feature_value, str) for feature_value in feature_values):
                        # this is a categorical feature
                        dict_name = cm_names[index] + '-' + str(chunk_tracker) + '-' + str(feature_number)
                        hist_info[dict_name] = {}
                        hist_info[dict_name]['classification'] = cm_names[index]
                        hist_info[dict_name]['chunk'] = chunk_tracker
                        hist_info[dict_name]['feature'] = feature_number
                        hist_info[dict_name]['counts'] = counts
                        hist_info[dict_name]['edges'] = values
                    else:
                        # this is a numerical feature
                        counts, edges = np.histogram(feature_values, bins=intervals)
                        dict_name = cm_names[index] + '-' + str(chunk_tracker) + '-' + str(feature_number)
                        hist_info[dict_name] = {}
                        hist_info[dict_name]['classification'] = cm_names[index]
                        hist_info[dict_name]['chunk'] = chunk_tracker
                        hist_info[dict_name]['feature'] = feature_number
                        hist_info[dict_name]['counts'] = counts
                        hist_info[dict_name]['edges'] = edges
                    index += 1
                feature_number += 1

            chunk_tracker += 1
            # empty bins
            TP, FP, FN, TN = [], [], [], []
            cm_values = [TP, FP, FN, TN]

        if chunk_tracker >= 2 and n_total_answers % comparison_block == 0 and drift_detection is None:
            # this means we can start comparing the densities
            first_chunk = chunk_tracker - 2
            second_chunk = chunk_tracker - 1
            vars()['comparison' + str(first_chunk)] = {}
            names = ['TP-', 'FP-', 'FN-', 'TN-']
            for classification in names:
                for feature in range(len(x)):
                    name_first = classification + str(first_chunk) + '-' + str(feature)
                    name_second = classification + str(second_chunk) + '-' + str(feature)
                    count_first_chunk = np.array(hist_info[name_first].get('counts') + prior)
                    count_second_chunk = np.array(hist_info[name_second].get('counts') + prior)
                    if len(count_first_chunk) == 0 or len(count_second_chunk) == 0:
                        # if any bin is empty, we cannot compare
                        continue
                    densities = count_second_chunk - count_first_chunk
                    densities = densities / count_first_chunk
                    subset = []
                    for bin in range(intervals):
                        subset.extend([densities[bin]] * count_first_chunk[bin])
                    name = classification + str(feature)
                    vars()['comparison' + str(first_chunk)][name] = {}
                    vars()['comparison' + str(first_chunk)][name]['subset'] = subset
            comparison = vars()['comparison' + str(first_chunk)]

            # compare density of TP and FN, and TN and FP
            for feature in range(len(x)):
                print('feature: ', feature)
                if 'TP-' + str(feature) in comparison and 'FN-' + str(feature) in comparison:
                    test = ranksums(comparison['TP-' + str(feature)].get('subset'),
                                    comparison['FN-' + str(feature)].get('subset'))
                    print('p value TP FN ', test.pvalue)
                    pvaluePos[feature].append(test.pvalue)
                    if test.pvalue < 0.01:
                        print(f'significance pos at {comparison_block*chunk_tracker}')
                if 'TN-' + str(feature) in comparison and 'FP-' + str(feature) in comparison:
                    test = ranksums(comparison['TN-' + str(feature)].get('subset'),
                                    comparison['FP-' + str(feature)].get('subset'))
                    if test.pvalue < 0.01:
                        print(f'significance neg at {comparison_block*chunk_tracker}')
                    print('p value TN FP ', test.pvalue)
                    pvalueNeg[feature].append(test.pvalue)

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
            if isinstance(dataset, PredictionInfluenceStream):
                print(dataset.weight)
            return metric

    return metric

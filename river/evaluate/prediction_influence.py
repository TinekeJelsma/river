import datetime as dt
import time
import typing
import numpy as np
from river import base
from river import metrics
from river import utils
from river import stream
from river.datasets.synth.prediction_influenced_stream import PredictionInfluenceStream
from scipy.stats import ranksums



__all__ = ['evaluate_influential']


def evaluate_influential(dataset: base.typing.Stream, model, metric: metrics.Metric,
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
    
    #def is_categorical(array_like):
    #    return array_like.dtype.name == 'category'

    preds = {}
    hist0 = []
    chunk_tracker = 0
    cm = metrics.ConfusionMatrix()
    TP, FP, FN, TN = [], [], [], []
    cm_values = [TP, FP, FN, TN]
    cm_names = ['TP', 'FP', 'FN', 'TN']
    hist_info = {}
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
            cm.update(y, y_pred)
            if isinstance(dataset, PredictionInfluenceStream):
                dataset.receive_feedback(y_true=y, y_pred=y_pred, x_features=x)
                # print(dataset.weight)
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

        model.learn_one(x=x, y=y)

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
            # fill all bins
            for feature in range(len(x)):
                index=0
                for cm_value in cm_values:
                    feature_values = [value.get(feature) for value in cm_value]                    
                    values, counts = np.unique(feature_values, return_counts=True) 
                    if all(isinstance(n, str) for n in feature_values) or len(counts) <10:
                        dict_name = cm_names[index] + '-' + str(chunk_tracker) + '-' + str(feature)
                        hist_info[dict_name] = {}
                        hist_info[dict_name]['classification'] = cm_names[index]
                        hist_info[dict_name]['chunk'] = chunk_tracker
                        hist_info[dict_name]['feature'] = feature
                        hist_info[dict_name]['counts'] = counts
                        hist_info[dict_name]['edges'] = values
                    else:
                        counts, edges = np.histogram(feature_values, bins = intervals)
                        dict_name = cm_names[index] + '-' + str(chunk_tracker) + '-' + str(feature)
                        hist_info[dict_name] = {}
                        hist_info[dict_name]['classification'] = cm_names[index]
                        hist_info[dict_name]['chunk'] = chunk_tracker
                        hist_info[dict_name]['feature'] = feature
                        hist_info[dict_name]['counts'] = counts
                        hist_info[dict_name]['edges'] = edges
                    index+=1

            chunk_tracker+=1
            # empty bins
            TP, FP, FN, TN = [], [], [], []
            cm_values = [TP, FP, FN, TN]

        
        if chunk_tracker >=2 and n_total_answers % comparison_block == 0:
            # this means we can start comparing the densities
            first_chunk = chunk_tracker-2
            second_chunk = chunk_tracker-1
            vars()['comparison' + str(first_chunk)] = {} 
            names = ['TP-', 'FP-', 'FN-', 'TN-']
            # calculate subset for TP
            for classification in names:
                for feature in range(len(x)):
                    name_first = classification + str(first_chunk) + '-' + str(feature)
                    name_second = classification + str(second_chunk) + '-' + str(feature)
                    count_first_chunk = np.array(hist_info[name_first].get('counts') + prior)
                    count_second_chunk = hist_info[name_second].get('counts') + prior
                    densities = np.array(count_second_chunk) - count_first_chunk
                    densities = densities/ count_first_chunk
                    subset = []
                    for bin in range(intervals):
                        subset.extend([densities[bin]]*count_first_chunk[bin])
                    name = classification + str(feature)
                    vars()['comparison' + str(first_chunk)][name] = {}
                    vars()['comparison' + str(first_chunk)][name]['subset'] = subset
            # print(vars()['comparison' + str(first_chunk)])
            comparison = vars()['comparison' + str(first_chunk)]

            # compare density of TP and FN
            for feature in range(len(x)):
                print('feature: ', feature)
                test = ranksums(comparison['TP-' + str(feature)].get('subset'), comparison['FN-' + str(feature)].get('subset'))
                print('p value TP FN ', test.pvalue)

                test = ranksums(comparison['TN-' + str(feature)].get('subset'), comparison['FP-' + str(feature)].get('subset'))
                print('p value TP FN ', test.pvalue)

        if n_total_answers > max_samples:
            print(cm)
            print(dataset.weight)
            # print("hist_info: ", hist_info)
            return metric

    return metric


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
import matplotlib.pyplot as plt
from river import drift

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

    # def is_categorical(array_like):
    #    return array_like.dtype.name == 'category'

    preds = {}
    chunk_tracker = 0
    cm = metrics.ConfusionMatrix()

    TP, FP, FN, TN = [], [], [], []
    cm_values = [TP, FP, FN, TN]
    cm_names = ['TP', 'FP', 'FN', 'TN']
    hist_info = {}
    pos_yvalues, pos_xvalues, neg_yvalues, neg_xvalues, xvalues, yvalues = [], [], [], [], [], []
    drift_detector_positive = drift.ADWIN()
    drift_detector_negative = drift.ADWIN()
    drift_detector = drift.ADWIN()
    pvaluePos = []
    pvalueNeg = []

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
            if y_pred == 1:
                for key, value in x.items():
                    drift_detector_positive.update(value)  # Data is processed one sample at a time
                    pos_yvalues.append(float(value))
                    pos_xvalues.append(n_total_answers)
                    if drift_detector_positive.change_detected:
                        # The drift detector indicates after each sample if there is a drift in the data
                        print(f'Change detected in positively classified at index {i} on feature {key}')
                        drift_detector_positive.reset()
                    # only check first feature for now
                    break
            if y_pred == 0:
                for key, value in x.items():
                    drift_detector_negative.update(value)  # Data is processed one sample at a time
                    neg_yvalues.append(float(value))
                    neg_xvalues.append(n_total_answers)
                    if drift_detector_negative.change_detected:
                        # The drift detector indicates after each sample if there is a drift in the data
                        print(f'Change detected  in negatively classified instances at index {i} on feature {key}')
                        drift_detector_negative.reset()
                    # only check first feature for now
                    break
            for key, value in x.items():
                drift_detector.update(value)  # Data is processed one sample at a time
                yvalues.append(float(value))
                xvalues.append(n_total_answers)
                if drift_detector_negative.change_detected:
                    # The drift detector indicates after each sample if there is a drift in the data
                    print(f'Change detected in all instances at index {i} on feature {key}')
                    drift_detector.reset()
                # only check first feature for now
                break

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
                        # plt.hist(feature_values, bins = intervals)
                        # plt.show()
                    index += 1
                feature_number += 1

            chunk_tracker += 1
            # empty bins
            TP, FP, FN, TN = [], [], [], []
            cm_values = [TP, FP, FN, TN]

        if chunk_tracker >= 2 and n_total_answers % comparison_block == 0:
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
                        continue
                    densities = count_second_chunk - count_first_chunk
                    densities = densities / count_first_chunk
                    subset = []
                    for bin in range(intervals):
                        subset.extend([densities[bin]] * count_first_chunk[bin])
                    name = classification + str(feature)
                    vars()['comparison' + str(first_chunk)][name] = {}
                    vars()['comparison' + str(first_chunk)][name]['subset'] = subset
            # print(vars()['comparison' + str(first_chunk)])
            comparison = vars()['comparison' + str(first_chunk)]

            # compare density of TP and FN, and TN and FP
            for feature in range(len(x)):
                print('feature: ', feature)
                if 'TP-' + str(feature) in comparison and 'FN-' + str(feature) in comparison:
                    test = ranksums(comparison['TP-' + str(feature)].get('subset'),
                                    comparison['FN-' + str(feature)].get('subset'))
                    print('p value TP FN ', test.pvalue)
                    pvaluePos.append(test.pvalue)
                    if test.pvalue < 0.01:
                        print(f'significance pos at {comparison_block*chunk_tracker}')
                if 'TN-' + str(feature) in comparison and 'FP-' + str(feature) in comparison:
                    test = ranksums(comparison['TN-' + str(feature)].get('subset'),
                                    comparison['FP-' + str(feature)].get('subset'))
                    if test.pvalue < 0.01:
                        print(f'significance neg at {comparison_block*chunk_tracker}')
                    print('p value TN FP ', test.pvalue)
                    pvalueNeg.append(test.pvalue)


            # visualize the distribution of data in hists:
            # for classification in names:
            #     for feature in range(len(x)):
            #         name_first = classification + str(first_chunk) + '-' + str(feature)
            #         name_second = classification + str(second_chunk) + '-' + str(feature)
            #         count_first_chunk = np.array(hist_info[name_first].get('counts') + prior)
            #         count_second_chunk = hist_info[name_second].get('counts') + prior
            #         edges_first_chunk = hist_info[name_first].get('edges')
            #         edges_second_chunk = hist_info[name_second].get('edges')
                    # fig, (ax1, ax2) = plt.subplots(1, 2)
                    # title = classification + "feature" + str(feature)
                    # # ax1.bar(edges_first_chunk[:-1], count_first_chunk, width=1, color='green')
                    # # ax2.bar(edges_second_chunk[:-1], count_second_chunk, width = 1, color='blue')
                    # # fig.suptitle(title)
                    # plt.suptitle(title)
                    # plt.bar(edges_first_chunk[:-1]-0.2, count_first_chunk, width=0.2, color='g')
                    # plt.bar(edges_second_chunk[:-1], count_second_chunk, width = 0.2, color='b')
                    # plt.show()
        if n_total_answers == max_samples:
            # plt.plot(pos_xvalues, pos_yvalues)
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 3))
            axes[0].plot(pos_xvalues, pos_yvalues)
            axes[1].plot(neg_xvalues, neg_yvalues)
            axes[2].plot(xvalues, yvalues)

            axes[0].title.set_text('Positively classified instances')
            axes[1].title.set_text('Negatively classified instances')
            axes[2].title.set_text('All instances')
            plt.show()
            plt.close()

            if isinstance(dataset, PredictionInfluenceStream):
                plt.plot(dataset.weight_tracker)
                plt.legend(['base negative', 'base positive', 'drift negative', 'drift positive', 'drift negative 2',
                            'drift positive 2'], loc=0)
                plt.show()

            # chunks size
            x = list(range(comparison_block*2, comparison_block*(chunk_tracker+1), comparison_block))
            plt.plot(x, pvalueNeg, label="p values for negative instances")
            plt.plot(x, pvaluePos, label='p values for positive instances')
            plt.legend()
            plt.show()

        if n_total_answers >= max_samples:
            print(cm)
            if isinstance(dataset, PredictionInfluenceStream):
                print(dataset.weight)
            # print("hist_info: ", hist_info)
            return metric

    return metric

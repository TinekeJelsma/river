from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.function_base import quantile
from river.base import DriftDetector
from river import metrics
from scipy.stats import bernoulli


def _compute_tpr(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])

def _compute_tnr(confusion_matrix):
    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])

def _compute_ppv(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

def _compute_npv(confusion_matrix):
    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])

METRICS_FUNCTION_MAPPING = {
    'tpr': _compute_tpr,
    'tnr': _compute_tnr,
    'ppv': _compute_ppv,
    'npv': _compute_npv
}

class LFR(DriftDetector):

    def __init__(self, bounds_table = None):
        super().__init__()
        # default values affected by init_bucket()
        self.time_decay = 0.9
        self.warn_level = 0.01
        self.detect_level = 0.00001
        self.metrics = {metric_name: PerformanceMetric(metric_name, self.time_decay)
                        for metric_name in ['tpr', 'tnr', 'ppv', 'npv']}
        
        self.confusion_matrix = metrics.ConfusionMatrix()
        self.confusion_matrix.update(1,1)
        self.confusion_matrix.update(0,0)
        self.confusion_matrix.update(1,0)
        self.confusion_matrix.update(0,1)
        self.idx = 0
        self.warnings = []
        self.detections = []
        self.warn_time = 0
        self.bounds_table = bounds_table
        self.concept_time_shifts = []
        # tpr; tnr; ppv; npv
    
    def update(self, y_true, y_pred):
        self.confusion_matrix.update(y_true, y_pred)
        # print(self.confusion_matrix)
        for metric in self.metrics.values():
            n, p_hat, r_hat = metric.update_metric(self.confusion_matrix, y_true, y_pred)
            # lb_warn, ub_warn, lb_detect, ub_detect = self.generate_bounds(n, p_hat,alpha_detect = self.detect_level, alpha_warn = self.warn_level)
            # # lb_detect, ub_detect = self.generate_bounds(n, p_hat, alpha=self.detect_level)
            # warn_shift = (r_hat <= lb_warn) or (r_hat >= ub_warn)
            # detect_shift = (r_hat <= lb_detect) or (r_hat >= ub_detect)

            # self.warnings.append(warn_shift)
            # self.detections.append(detect_shift)

            # print("Sample %i: metric %s, R: %.3f, Warn LB: %.3f Warn UB: %.3f, Detect LB: %.3f, Detect UB: %.3f, warn: %s detect: %s"
            #           % (self.idx, metric.metric_name, r_hat, lb_warn, ub_warn, lb_detect, ub_detect, warn_shift, detect_shift))

            # if any(self.warnings) and self.warn_time is None:
            #     self.warn_time = self.idx
            # elif all([not warning for warning in self.warnings]) and self.warn_time is not None:
            #     self.warn_time = None
            
            # if any(self.detections) and self.idx >50:
            #     self.detections = []
            #     self.concept_time_shifts.append(self.idx)
            #     for metric in self.metrics.values():
            #         metric.reset_internals()
            #     self.confusion_matrix.reset()
            #     self.confusion_matrix.update(1,1)
            #     self.confusion_matrix.update(0,0)
            #     self.confusion_matrix.update(1,0)
            #     self.confusion_matrix.update(0,1)
        self.idx += 1
           
    def generate_bounds(self, n, p_hat, n_sim=1000, alpha_detect = 0.00001, alpha_warn = 0.01):
        n = int(n)
        R = [None] * n_sim
        for j in range(n_sim):
            summation = 0
            bernoulli_samples = bernoulli.rvs(p_hat, size = n)
            for i in range(n):
                summation +=(pow(self.time_decay,(n - i -1))) * bernoulli_samples[i]
            R[j] = summation* (1- self.time_decay)
        lb_detect, ub_detect = np.percentile(R, q=[alpha_detect, (1 - alpha_detect)])
        lb_warn, ub_warn = np.percentile(R, q=[alpha_warn, (1 - alpha_warn)])

        return lb_warn, ub_warn, lb_detect, ub_detect
    
    def show_metric(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        for metric in self.metrics.values():
            axes[0].plot(metric._P, label = f'P {metric.metric_name}')
            axes[0].title.set_text('P rate throughout time')
            axes[0].legend()
            # print(f'{metric.metric_name} P: {metric._P}')
            # print(f'{metric.metric_name} R: {metric._R}')
        for metric in self.metrics.values():
            axes[1].plot(metric._R, label = f'R {metric.metric_name}')
            axes[1].title.set_text('R rate throughout time')
            axes[1].legend()
        plt.show()



class PerformanceMetric(object):

    def __init__(self, metric_name, time_decay):
        if metric_name not in ['tpr', 'tnr', 'ppv', 'npv']:
            raise ValueError('metric_name must be one of tpr, tnr, ppv, or npv, got %s' % metric_name)

        self.metric_name = metric_name
        self.time_decay = time_decay
        self.metric_value = [0.5]
        self._R = [0.5]
        self._P = [0.5]
    
    def reset_internals(self):
        self._R[-1] = 0.5
        self._P[-1] = 0.5
    
    def metric_influenced(self):
        if len(self.metric_value) > 1:
            return abs(self.metric_value[-1] - self.metric_value[-2]) > 0
        elif len(self.metric_value) < 2:
            return 1
        else:
            return 0

    def update_metric(self, confusion_matrix, y_true, y_pred):
        # if y true = pos, y pred =pos,  influence tpr and ppv
        # if y true = neg, ypred= neg, influence tnr, and npv
        # if y_true = pos, ypred = neg, influence tnr and ppv
        # if y_true = neg, y_pred = pos, influence tpr and npv
        self.metric_value.append(METRICS_FUNCTION_MAPPING[self.metric_name](confusion_matrix))

        if self.metric_influenced():
            self._R.append(self.time_decay * self._R[-1] + (1 - self.time_decay) * int(y_true == y_pred))
        else:
            self._R.append(self._R[-1])
        if self.metric_name == 'tpr':
            n = confusion_matrix[1][1] + confusion_matrix[1][0]
            self._P.append(confusion_matrix[1][1]/ n)
        if self.metric_name == 'tnr':
            n = confusion_matrix[0][0] + confusion_matrix[0][1]
            self._P.append(confusion_matrix[0][0]/ n)
        if self.metric_name == 'ppv':
            n = confusion_matrix[0][1] + confusion_matrix[1][1]
            self._P.append(confusion_matrix[1][1]/ n)
        if self.metric_name == 'npv':
            n = confusion_matrix[0][0] + confusion_matrix[1][0]
            self._P.append(confusion_matrix[0][0]/ n)

        return n, self._P[-1], self._R[-1]
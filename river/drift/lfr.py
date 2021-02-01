from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.function_base import quantile
from river.base import DriftDetector
from river import metrics
from scipy.stats import bernoulli
import os
import pickle

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

    def __init__(self, time_decay = 0.9, warn_level = 0.01, detect_level = 0.00001, max_samples: int = None, burn_in: int = 50):
        super().__init__()
        # default values affected by init_bucket()
        self.time_decay = time_decay
        self.warn_level = warn_level
        self.detect_level = detect_level
        self.max_samples = max_samples
        self.metrics = {metric_name: PerformanceMetric(metric_name, self.time_decay)
                        for metric_name in ['tpr', 'tnr', 'ppv', 'npv']}
        
        self.confusion_matrix = metrics.ConfusionMatrix()
        self.reset_confusion_matrix()
        self.burn_in = burn_in
        self.idx = 0
        self.warnings = []
        self.detections = []
        self.warn_time = 0
        self.bounds_table = BoundTable(self.time_decay, self.warn_level, self.detect_level, self.max_samples)
        self.concept_time_shifts = []
        self.after_detection = 0
        self.first_detection = False
    
    def reset_confusion_matrix(self):
        self.confusion_matrix.reset()
        self.confusion_matrix.update(1,1)
        self.confusion_matrix.update(0,0)
        self.confusion_matrix.update(1,0)
        self.confusion_matrix.update(0,1)
    
    def update(self, y_true, y_pred):
        self.confusion_matrix.update(y_true, y_pred)
        # print(self.confusion_matrix)
        for metric in self.metrics.values():
            n, p_hat, r_hat = metric.update_metric(self.confusion_matrix, y_true, y_pred)
            lb_warn, ub_warn, lb_detect, ub_detect = self.bounds_table.get_bounds(n, p_hat)
            # lb_detect, ub_detect = self.generate_bounds(n, p_hat, alpha=self.detect_level)
            warn_shift = (r_hat <= lb_warn) or (r_hat >= ub_warn)
            detect_shift = (r_hat <= lb_detect) or (r_hat >= ub_detect)

            self.warnings.append(warn_shift)
            self.detections.append(detect_shift)

            print("Sample %i: metric %s, R: %.3f, Warn LB: %.3f Warn UB: %.3f, Detect LB: %.3f, Detect UB: %.3f, warn: %s detect: %s"
                      % (self.idx, metric.metric_name, r_hat, lb_warn, ub_warn, lb_detect, ub_detect, warn_shift, detect_shift))

            if any(self.warnings) and self.warn_time is None:
                self.warn_time = self.idx
            elif all([not warning for warning in self.warnings]) and self.warn_time is not None:
                self.warn_time = None
            
            if any(self.detections) and self.idx > self.burn_in:
                self.detections = []
                if self.concept_time_shifts:
                    if self.concept_time_shifts[-1] + self.burn_in < self.idx:
                        self.concept_time_shifts.append(self.idx)
                        for metric in self.metrics.values():
                            metric.reset_internals()
                            self.reset_confusion_matrix()
                if not self.concept_time_shifts:
                    self.concept_time_shifts.append(self.idx)
                    for metric in self.metrics.values():
                        metric.reset_internals()
                        self.reset_confusion_matrix()
        self.idx += 1
    
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

class BoundTable(object):
    def __init__(self, time_decay, warn_level, detect_level, max_samples) -> None:
        super().__init__() 
        self.time_decay = time_decay
        self.warn_level = warn_level
        self.detect_level = detect_level
        self.max_samples = max_samples
        steps = 0.01
        self.p_range = np.arange(0, 1 + steps, steps)
        self.n_range = range(1, int(self.max_samples/4), 1)
        self.filename  = str(f'warn{self.warn_level}-detect{self.detect_level}-psteps{steps}')
    
    def get_path(self):
        return os.path.join(os.getcwd(), f'{self.filename}.pkl')

    def table_exists(self):
        return os.path.isfile(self.get_path())
    
    def create_table(self):
        bound_dict = {}
        for n in self.n_range:
            bound_dict[n] = {}
            for p_hat in self.p_range:
                p_hat = round(p_hat, 3)
                bound_dict[n][p_hat] = {}
                lb_warn, ub_warn, lb_detect, ub_detect = self.generate_bounds(n, p_hat)
                bound_dict[n][p_hat] = {'lb_warn': lb_warn, 'ub_warn': ub_warn, 'lb_detect': lb_detect, 'ub_detect': ub_detect}
        self.save_table(bound_dict)

        return bound_dict

    def save_table(self, bound_dict):
        with open(f'{self.filename}.pkl', 'wb') as pickleFile:
            pickle.dump(bound_dict, pickleFile)
            pickleFile.close()

    def update_table(self, bound_dict, n, p_hat):
        bound_dict[n] = {}
        for p_hat in self.p_range:
            p_hat = round(p_hat, 3)
            bound_dict[n][p_hat] = {}
            lb_warn, ub_warn, lb_detect, ub_detect = self.generate_bounds(n, p_hat)
            bound_dict[n][p_hat] = {'lb_warn': lb_warn, 'ub_warn': ub_warn, 'lb_detect': lb_detect, 'ub_detect': ub_detect}
        self.save_table(bound_dict)

        return bound_dict

    def generate_bounds(self, n, p_hat, n_sim=1000):
        n = int(n)
        R = [None] * n_sim
        for j in range(n_sim):
            summation = 0
            bernoulli_samples = bernoulli.rvs(p_hat, size = n)
            for i in range(n):
                summation +=(pow(self.time_decay,(n - i -1))) * bernoulli_samples[i]
            R[j] = summation* (1- self.time_decay)
        lb_detect, ub_detect = np.quantile(R, q=[self.detect_level, (1 - self.detect_level)])
        lb_warn, ub_warn = np.quantile(R, q=[self.warn_level, (1 - self.warn_level)])

        return lb_warn, ub_warn, lb_detect, ub_detect
    
    def get_bounds(self, n, p_hat):
        n = int(n)
        p_hat = round(p_hat, 1)
        if self.table_exists():
            bound_dict = pickle.load(open(f'{self.filename}.pkl', 'rb'))
        else:
            bound_dict = self.create_table()
        # print(f'bound table: {bound_dict}')
        # print(f'N = {n} p_hat = {p_hat}')
        if n not in bound_dict:
            bound_dict = self.update_table(bound_dict, n, p_hat)
        ub_detect = bound_dict[n][p_hat].get('ub_detect')
        lb_detect = bound_dict[n][p_hat].get('lb_detect')
        ub_warn = bound_dict[n][p_hat].get('ub_warn')
        lb_warn = bound_dict[n][p_hat].get('lb_warn')

        return lb_warn, ub_warn, lb_detect, ub_detect
        

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
import textwrap

import numpy as np

from .. import base
from ..synth import Agrawal
from river.utils.skmultiflow_utils import check_random_state
import random


class PredictionInfluenceStream(base.SyntheticDataset):

    def __init__(self, stream= [Agrawal(seed=112), 
                                Agrawal(seed=112,
                                classification_function=2)],
                 seed: int = None,
                 weight: list =None,
                 weight_correct=1.001,
                 weight_incorrect=0.999,
                 influence_method="multiplication"):
        # Fairly simple check for consistent number of features
        if stream[0].n_features != stream[1].n_features:
            raise AttributeError(f"Inconsistent number of features between "
                                 f"{stream.__name__} ({stream[0].n_features}) and "
                                 f"{stream.__name__} ({stream[1].n_features}).")
        super().__init__(n_features=stream[0].n_features, n_classes=stream[0].n_classes,
                         n_outputs=stream[0].n_outputs, task=stream[0].task, n_samples = stream[0].n_samples)

        self.stream = stream
        self.weight = weight
        self.weight_tracker = []
        self.weight_tracker_dynamic = []
        self.last_stream = None
        self.weight_correct = weight_correct
        self.weight_incorrect = weight_incorrect
        self.cache = []
        self.influence_method = influence_method
        self.n_streams = len(stream)
        self.seed = seed
        self.source_stream = []

        self.set_weight()
        self.set_influence_method()


    def set_weight(self):
        if self.weight is None:
            counter = len(self.stream)
            start = [1] * counter
            self.weight = [1] * counter
        self.weight_tracker = [start]
        self.weight_tracker_dynamic = [start]

    def set_influence_method(self):
        if self.influence_method != "multiplication" and self.influence_method != "addition":
            self.influence_method = "multiplication"

    def __iter__(self):
        rng = check_random_state(self.seed)
        sample_idx = 0
        n_streams = list(range(self.n_streams))
        normalized_weights = [float(i) / max(self.weight) for i in self.weight]
        instance_generator = []
        for i in range(self.n_streams):
            instance_generator.append(iter(self.stream[i]))

        while True:
            sample_idx += 1
            probability = random.choices(n_streams, normalized_weights)
            current_stream = probability[0]
            self.source_stream.append(current_stream)
            try:
                x, y = next(instance_generator[current_stream])
            except StopIteration:
                break
            yield x, y
    
    def receive_feedback(self, y_true, y_pred, x_features):
        if isinstance(y_true, int):
            # and isinstance(y_pred, int)
            y_true, y_pred, x_features = [y_true], [y_pred], [x_features]
        for i in range(len(y_true)):
            if y_true[i] is not None:
                if len(self.cache) == 0 or (y_pred[i] == self.cache[0][0] and x_features[i] == self.cache[0][1]):
                    self.receive_feedback_update(y_true[i], y_pred[i], self.source_stream[i])
                    if len(self.cache) != 0:
                        self.cache.remove(self.cache[0])
                        while len(self.cache[0]) == 4:
                            self.receive_feedback_update(y_true[i], y_pred[i], self.source_stream[i])
                            self.cache.remove(self.cache[0])
                else:
                    wait_for_feedback = [y_pred[i], x_features[i], y_true[i], self.source_stream[i]]
                    self.cache.append(wait_for_feedback)
            else:
                no_label = [y_pred[i], x_features[i], self.source_stream[i]]
                self.cache.append(no_label)
        self.source_stream = []

    def receive_feedback_update(self, y_true, y_pred, stream):
        if y_true == y_pred:
            if self.influence_method == "multiplication":
                self.weight[stream] = self.weight[stream] * self.weight_correct
            else:
                self.weight[stream] = self.weight[stream] + self.weight_correct
        else:
            if self.influence_method == "multiplication":
                self.weight[stream] = self.weight[stream] * self.weight_incorrect
            else:
                self.weight[stream] = self.weight[stream] + self.weight_incorrect
        self.weight_tracker_dynamic.append(self.weight.copy())
        #print(self.weight_tracker_dynamic)


    def __repr__(self):
        params = self._get_params()
        l_len_config = max(map(len, params.keys()))
        r_len_config = max(map(len, map(str, params.values())))

        config = '\n\nConfiguration:\n'
        for k, v in params.items():
            if not isinstance(v, base.SyntheticDataset):
                indent = 0
            else:
                indent = l_len_config + 2
            config += ''.join(k.rjust(l_len_config) + '  ' +
                              textwrap.indent(str(v).ljust(r_len_config), ' ' * indent)) + '\n'

        l_len_prop = max(map(len, self._repr_content.keys()))
        r_len_prop = max(map(len, self._repr_content.values()))

        out = (
                f'Synthetic data generator\n\n' +
                '\n'.join(
                    k.rjust(l_len_prop) + '  ' + v.ljust(r_len_prop)
                    for k, v in self._repr_content.items()
                ) + config
        )

        return out

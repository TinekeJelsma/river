import textwrap

import numpy as np

from .. import base
from ..synth import Agrawal
from river.utils.skmultiflow_utils import check_random_state
import random
from itertools import chain


class PredictionInfluenceStream(base.SyntheticDataset):

    def __init__(self, stream=[Agrawal(seed=112),
                               Agrawal(seed=112,
                                       classification_function=2)],
                 seed: int = None,
                 weight: list = None,
                 weight_correct=0.99,
                 weight_incorrect=1.01,
                 influence_method="multiplication",
                 weight_update: int = 1,
                 weight_update_delay: int = 1):
        # Fairly simple check for consistent number of features
        if len(stream) > 1 and stream[0].n_features != stream[1].n_features:
            raise AttributeError(f"Inconsistent number of features between "
                                 f"{stream.__name__} ({stream[0].n_features}) and "
                                 f"{stream.__name__} ({stream[1].n_features}).")
        super().__init__(n_features=stream[0].n_features, n_classes=stream[0].n_classes,
                         n_outputs=stream[0].n_outputs, task=stream[0].task, n_samples=stream[0].n_samples)
        if hasattr(stream[0], 'feature_names'):
            self.feature_names = stream[0].feature_names
        self.stream = stream
        self.current_stream_weight = []
        self.stream_weight = weight
        self.start_weight = weight
        self.weight_update = weight_update
        self.weight_tracker = []
        self.weight_update_delay = weight_update_delay
        self.last_stream = None
        self.weight_correct = weight_correct
        self.weight_incorrect = weight_incorrect
        self.cache = []
        self.influence_method = influence_method
        self.n_streams = len(stream)
        self.n_features = stream[0].n_features
        self.seed = seed
        self.source_stream = []
        self.idx = 0

        self.set_weight()
        self.set_influence_method()

    def set_weight(self):
        if self.stream_weight is None:
            counter = len(self.stream)
            self.stream_weight = [1] * counter
        self.current_stream_weight = self.stream_weight.copy()
        self.weight_tracker = [self.stream_weight.copy()]

    def set_influence_method(self):
        if self.influence_method != "multiplication" and self.influence_method != "addition":
            self.influence_method = "multiplication"

    def __iter__(self):
        # This function decides from which stream the next sample should be chosen.
        rng = check_random_state(self.seed)
        sample_idx = 0
        n_streams = list(range(self.n_streams))
        instance_generator = []
        for i in range(self.n_streams):
            instance_generator.append(iter(self.stream[i]))

        while True:
            # normalize all weights and then choose the stream for new instance.
            if len(self.weight_tracker) > self.weight_update_delay:
                normalized_weights = [float(i) / max(self.weight_tracker[-self.weight_update_delay]) for i in self.weight_tracker[-self.weight_update_delay]]
            else:
                normalized_weights = [float(i) / max(self.weight_tracker[0]) for i in self.weight_tracker[0]]
            probability = random.choices(n_streams, normalized_weights)
            current_stream = probability[0]
            self.source_stream.append(current_stream)
            sample_idx += 1
            try:
                x, y = next(instance_generator[current_stream])
            except StopIteration:
                break
            yield x, y

    def receive_feedback(self, y_true, y_pred, x_features):
        if isinstance(y_true, int):
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
                self.current_stream_weight[stream] = self.current_stream_weight[stream] * self.weight_correct
            else:
                self.current_stream_weight[stream] = self.current_stream_weight[stream] + self.weight_correct
        else:
            if self.influence_method == "multiplication":
                self.current_stream_weight[stream] = self.current_stream_weight[stream] * self.weight_incorrect
            else:
                self.current_stream_weight[stream] = self.current_stream_weight[stream] + self.weight_incorrect
        self.weight_tracker.append(self.stream_weight.copy())
        # if weight of streams should be updated, then set the stream_weight to curren_stream_weight
        if self.idx % self.weight_update == 0 and self.idx > 0:
            self.stream_weight = self.current_stream_weight.copy()
        self.idx += 1
        if any((x > 0 and x < 0.2) for x in self.stream_weight) and self.idx > 50:
            # if there is a stream with a weight between 0 and 0.2 and the classifier has 
            # classified more than 50 instance, we can add a new concept which starts with a weight of 0.5
            self.add_concept(border = 0.2, new = 0.5)

    def add_concept(self, border=0.01, new=0.1):
        for stream in range(self.n_streams):
            # the new concept stream is always the stream that is 2 places further and should not be introduced yet (currently weight of 0)
            if self.current_stream_weight[stream] < border and self.current_stream_weight[stream] > 0 and (stream + 2) < self.n_streams:
                if self.current_stream_weight[stream + 2] == 0:
                    self.current_stream_weight[stream + 2] = new
            elif self.current_stream_weight[stream] < border and self.current_stream_weight[stream] > 0 and (stream + 2) >= self.n_streams:
                if self.current_stream_weight[stream - self.n_streams + 2] == 0:
                    stream_number = stream - self.n_streams + 2
                    self.current_stream_weight[stream_number] = new

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

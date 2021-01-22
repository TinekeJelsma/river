from river import synth
from river import preprocessing
from river import evaluate
from river import metrics
from river import datasets
from river import tree
from river import compose
from river import optim
from river import datasets
from river import linear_model
from river.drift import LFR
from river import neighbors

import itertools

lfr_metric = LFR()
streams = []

for x in range(1):
    # base negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # base positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))
    # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))
    # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))

# X_y = synth.PredictionInfluenceStream(stream= streams, weight_incorrect=1.01, weight_correct=0.99, weight_update = 1, weight = [1,1,0,0,0,0])

gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
                                drift_stream=synth.SEA(seed=42, variant=1),
                                seed=1, position=2000, width=1000)
#  Take 1000 instances from the infinite data generator
 
X_y = iter(gen.take(20000))
metric = metrics.Accuracy()

# model = tree.HoeffdingAdaptiveTreeClassifier(
#     grace_period=100,
#     split_confidence=1e-5,
#     leaf_prediction='nb',
#     nb_threshold=10,
#     seed=0
# )

model = linear_model.ALMAClassifier()


evaluate.evaluate_influential(X_y, model, max_samples = 10000, metric = metric, print_every=50, drift_detection= lfr_metric)
print(f' time shifts are: {lfr_metric.concept_time_shifts}')
lfr_metric.show_metric()

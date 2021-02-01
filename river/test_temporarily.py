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
import itertools

streams = []

for x in range(1):
    # base negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[1, 0]))
    # base positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[0, 1]))
    # drift negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[1, 0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[0, 1]))
    # drift negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[1, 0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1,
                                        class_weights=[0, 1]))

# self defeating influence
X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=1.02, weight_correct=0.98, weight_update=1,
                                      weight=[1, 1, 0, 0, 0, 0])

# self fulfilling influence
# X_y = synth.PredictionInfluenceStream(stream= streams, weight_incorrect=0.99, weight_correct=1.01, weight_update = 1)

model = preprocessing.StandardScaler()
model |= linear_model.ALMAClassifier()
metric = metrics.Accuracy()
evaluate.evaluate_influential(X_y, model, metric, print_every=10000, comparison_block=100, intervals=8,
                              max_samples=5000, prior=1)

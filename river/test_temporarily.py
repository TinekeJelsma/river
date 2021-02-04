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
from river import compat
import itertools
from sklearn import linear_model

max_samples = 1500
n_features = 5
n_centroids = 5
streams = []

for x in range(1):
    # base negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # base positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))
    # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))
    # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))

X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=1.02, weight_correct=0.98, weight_update=1,
                                      weight=[1, 1,0,0,0,0])
# X_y = datasets.LendingClub()
# max_samples=15384462
# for x, y in X_y.take(5):
#     print(x, y)


model = preprocessing.StandardScaler()
model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])
hidden_model = preprocessing.StandardScaler()
hidden_model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])

metric = metrics.Accuracy()
hidden_metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model= model, hidden_model = hidden_model, metric = metric, hidden_metric = hidden_metric, print_every=100, comparison_block = 1000, intervals = 8, max_samples=5000, prior = 1)

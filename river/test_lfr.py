from river import synth
from river import preprocessing
from river import evaluate
from river import metrics
from river import datasets
from river import tree
from river import compose
from river import optim
from river import datasets
# from river import linear_model
from sklearn import linear_model
from sklearn import tree
from river.drift import LFR
from river import neighbors
from river import compat
import itertools
from river import compose

streams = []
max_samples = 500
lfr_metric = LFR(max_samples=max_samples, burn_in=50)
n_features = 5
n_centroids = 5

for x in range(3):
    # negative
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # positive
    streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))

X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=1.02, weight_correct=0.98, weight_update=1,
                                      weight=[1, 1, 0, 0, 0, 0])

metric = metrics.Accuracy()

model = preprocessing.StandardScaler()

model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])


evaluate.evaluate_influential(X_y, model, max_samples=max_samples, metric=metric, print_every=50,
                              drift_detection=lfr_metric, batch_size=1)

print(f' time shifts are: {lfr_metric.concept_time_shifts}')

lfr_metric.show_metric()

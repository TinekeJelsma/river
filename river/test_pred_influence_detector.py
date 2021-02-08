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
from sklearn import linear_model
from river import compat
import itertools

streams = []
max_samples = 10000
n_features = 1
n_centroids = 1

for x in range(5):
    # negative
    streams.append(synth.RandomRBFDrift(seed_model = 123 + x, n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # positive
    streams.append(synth.RandomRBFDrift(seed_model = 225 + x, n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))

X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=1.01, weight_correct=0.99, weight_update=1, weight_update_delay=100, weight=[1,1,0,0,0,0,0,0,0,0])
model = preprocessing.StandardScaler()
model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])

metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model= model, metric = metric, print_every=100, comparison_block = 1000, max_samples=max_samples, prior = 1, batch_size=1)

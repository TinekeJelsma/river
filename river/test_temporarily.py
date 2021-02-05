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
max_samples = 5000
n_features = 2
n_centroids = 5

for x in range(5):
    # base negative
    streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[1, 0]))
    # base positive
    streams.append(synth.RandomRBFDrift(seed_model=60+x, seed_sample=30+x, n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        class_weights=[0, 1]))
    # # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        # class_weights=[1, 0]))
    # drift positive
    # streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        # class_weights=[0, 1]))
    # drift negative
    # streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        # class_weights=[1, 0]))
    # drift positive
    # streams.append(synth.RandomRBFDrift(n_classes=2, n_features=n_features, change_speed=0, n_drift_centroids=1, n_centroids=n_centroids,
                                        # class_weights=[0, 1]))

X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=1.01, weight_correct=0.99, weight_update=1)
# weight=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
model = preprocessing.StandardScaler()
model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])
hidden_model = preprocessing.StandardScaler()
hidden_model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])

metric = metrics.Accuracy()
hidden_metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model= model, hidden_model = hidden_model, metric = metric, hidden_metric = hidden_metric,
 print_every=100, comparison_block = 1000, intervals = 8, max_samples=max_samples, prior = 1, batch_size=100, hidden_batch_size=100)

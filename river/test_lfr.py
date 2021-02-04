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
max_samples = 1500
lfr_metric = LFR(max_samples=max_samples, burn_in=50)
hidden_lfr = LFR(max_samples=max_samples, burn_in = 50)
n_features = 5
n_centroids = 5

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

X_y = synth.PredictionInfluenceStream(stream=streams, weight_incorrect=0.98, weight_correct=1.02, weight_update=1,
                                      weight=[1, 1, 1, 1, 1, 1])

# gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
# drift_stream=synth.SEA(seed=42, variant=1),
# seed=1, position=100, width=50)
#  Take 1000 instances from the infinite data generator
# gen = datasets.LendingClub()
# X_y = iter(gen.take(max_samples))

metric = metrics.Accuracy()
hidden_metric = metrics.Accuracy()
# model = tree.HoeffdingAdaptiveTreeClassifier(
#     grace_period=100,
#     split_confidence=1e-5,
#     leaf_prediction='nb',
#     nb_threshold=10,
#     seed=0
# )
model = preprocessing.StandardScaler()
# model |= linear_model.ALMAClassifier()

model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])
hidden_model = preprocessing.StandardScaler()
hidden_model |= compat.convert_sklearn_to_river(
    estimator=linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='optimal'), classes=[0, 1])


evaluate.evaluate_influential(X_y, model, max_samples=max_samples, metric=metric, hidden_metric = hidden_metric, print_every=50,
                              drift_detection=lfr_metric, hidden_drift_detection = hidden_lfr, batch_size=1, hidden_batch_size=1, hidden_model = hidden_model)
print(f' time shifts are: {lfr_metric.concept_time_shifts}')
print(f' hidden time shifts are: {hidden_lfr.concept_time_shifts}')

lfr_metric.show_metric()
hidden_lfr.show_metric()

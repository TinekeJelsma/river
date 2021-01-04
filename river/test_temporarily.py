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
    streams.append(synth.RandomRBFDrift(seed_model=30, seed_sample=12, n_classes=2, n_features=1, change_speed=1, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # base positive
    streams.append(synth.RandomRBFDrift(seed_model=50, seed_sample=78, n_classes=2, n_features=1, change_speed=1, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))
    # drift negative
    streams.append(synth.RandomRBFDrift(seed_model=54, seed_sample=68, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(seed_model=56, seed_sample=23, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))
    # drift negative
    streams.append(synth.RandomRBFDrift(seed_model=52, seed_sample=68, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[1,0]))
    # drift positive
    streams.append(synth.RandomRBFDrift(seed_model=59, seed_sample=23, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1, class_weights=[0,1]))

# for x in range(6):
#     streams.append(synth.Agrawal(classification_function=x, seed=42))
# set1 = synth.RandomRBFDrift(seed_model=34, seed_sample=30, n_classes=2, n_features=1, change_speed=0, n_drift_centroids=1, n_centroids=1)
# set2 = synth.RandomRBFDrift(seed_model=34, seed_sample=30, n_classes=2, n_features=1, change_speed=0.5, n_drift_centroids=1, n_centroids=1)
# X_y = {}

# X_y = itertools.chain(set1.take(1000), set2.take(1000))
X_y = synth.PredictionInfluenceStream(stream= streams, weight_incorrect=1.02, weight_correct=0.95, weight_update = 1, weight = [1,1,0,0,0,0])

# X_y = synth.RandomRBFDrift(seed_model=30, seed_sample=30, n_classes=2, n_features=4, change_speed=0, n_drift_centroids=10, n_centroids=20, class_weights=[0.5,0,5])
# X_y = datasets.CreditCard()
# for x, y in X_y.take(5):
#     print(x, y)
# model = preprocessing.StandardScaler()
# model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, split_confidence=1e-5, leaf_prediction='nb', nb_threshold=10, seed=0)

model = preprocessing.StandardScaler() 
model |= linear_model.ALMAClassifier()
metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model, metric, print_every=1000, comparison_block = 2000, intervals = 8, max_samples=5000, prior = 1)
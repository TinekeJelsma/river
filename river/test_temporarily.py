
from river import synth
from river import preprocessing
from river import evaluate
from river import metrics
from river import datasets
from river import tree
from river import compose
from river import optim

streams = []

# for x in range(5):
#     streams.append(synth.RandomRBFDrift(seed_model=30+x, seed_sample=30, n_classes=2, n_features=4, change_speed=0.87, n_drift_centroids=10, n_centroids=20, class_weights=[0,1]))
#     streams.append(synth.RandomRBFDrift(seed_model=50+x, seed_sample=30, n_classes=2, n_features=4, change_speed=0.87, n_drift_centroids=10, n_centroids=20, class_weights=[1,0]))

for x in range(6):
    streams.append(synth.Agrawal(classification_function=x, seed=42))

X_y = synth.PredictionInfluenceStream(stream= streams)

# model = preprocessing.StandardScaler()
model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, split_confidence=1e-5, leaf_prediction='nb', nb_threshold=10, seed=0)

metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model, metric, print_every=100, comparison_block = 200, intervals = 4, max_samples=400, prior = 1)
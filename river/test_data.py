from river import synth
from river import preprocessing
from river import evaluate
from river import metrics
from river import datasets
from river import tree
from river import compose
from river import optim

X_y = synth.PredictionInfluenceStream(
         stream= [synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=2, n_features=4, n_centroids=20), 
         synth.RandomRBF(seed_model=41, seed_sample=49, n_classes=2, n_features=4, n_centroids=20)])

model = preprocessing.StandardScaler()
model |= tree.HoeffdingAdaptiveTreeClassifier(grace_period=100, split_confidence=1e-5, leaf_prediction='nb', nb_threshold=10, seed=0)

metric = metrics.Accuracy()

evaluate.evaluate_influential(X_y, model, metric, print_every=100, comparison_block = 100, intervals = 4, max_samples=200, prior = 1)
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
gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
                                drift_stream=synth.SEA(seed=42, variant=1),
                                seed=1, position=50, width=50)
 # Take 1000 instances from the infinite data generator
dataset = iter(gen.take(100))
metric = metrics.Accuracy()

model = tree.HoeffdingAdaptiveTreeClassifier(
    grace_period=100,
    split_confidence=1e-5,
    leaf_prediction='nb',
    nb_threshold=10,
    seed=0
)

evaluate.evaluate_lfr(dataset, model, metric = metric, print_every=50, lfr = lfr_metric)
print(f' time shifts are: {lfr_metric.concept_time_shifts}')
lfr_metric.show_metric()

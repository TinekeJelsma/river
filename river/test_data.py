from river import synth
from river import linear_model
from river import preprocessing
from river import evaluate
from river import metrics
from river import datasets

def demo():
    dataset = synth.PredictionInfluenceStream(
        stream= [synth.RandomRBF(seed_model=42, seed_sample=42, n_classes=2, n_features=4, n_centroids=20), 
        synth.RandomRBF(seed_model=41, seed_sample=49, n_classes=2, n_features=4, n_centroids=20)])

    stream = dataset.take(1000)
    
    model = compose.Pipeline(('scale', preprocessing.StandardScaler()), ('lin_reg', linear_model.LinearRegression())
    evaluate.progressive_val_score(model=model, dataset=datasets.Phishing(),metric=metrics.ROCAUC(),print_every=200)


if __name__ == '__main__':
    demo()

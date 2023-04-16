import pandas as pd
from modules.neural_network import NeuralNetwork

# import data
train = pd.read_csv("data/mnist_test.csv")
test = pd.read_csv("data/mnist_train.csv")

nn = NeuralNetwork()
nn.fit(
    X=train.drop(columns="label").to_numpy(),
    y=train.label.to_numpy(),
    hiddenNodes=[5, 3, 4],
    stepSize=0.1,
    batchSize=100,
    ITERS=100,
    seed=0,
)

# evaluate on test data
preds = nn.predict(X=test.drop(columns="label").to_numpy())
(preds == test.label).mean()

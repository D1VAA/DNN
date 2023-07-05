from NeuralNetwork.mlp import MultiLayerPerceptron
from sklearn.datasets import load_digits

digits = load_digits()
target = digits.target

n = len(digits.images)

images = digits.images.reshape((n, -1))

nn = MultiLayerPerceptron(images, target, [10], epoch=50, eta=0.01)
nn.train()

from DNN import NeuralNetwork
from sklearn.datasets import load_digits

digits = load_digits()
target = digits.target

n = len(digits.images)

images = digits.images.reshape((n, -1))

nn = NeuralNetwork()
nn.training_config(images, target, [20, 20])
nn.train()

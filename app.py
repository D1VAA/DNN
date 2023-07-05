from DNN import NeuralNetwork
from sklearn.datasets import load_digits

digits = load_digits()
target = digits.target

n = len(digits.images)

images = digits.images.reshape((n, -1))

nn = NeuralNetwork()
nn.network_config(images, target, [30], epoch=50, eta=0.01)
nn.train()

from DNN import NeuralNetwork
from sklearn.datasets import load_digits

digits = load_digits()

n = len(digits.images)

images = digits.images.reshape((n, -1))

nn = NeuralNetwork()
nn.training_config(images, [20, 20])
nn2 = nn.train()

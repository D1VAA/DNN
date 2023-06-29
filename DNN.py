import numpy as np
import pandas as pd
from math import exp
from sklearn.datasets import load_digits

digits = load_digits()

n = len(digits.images)

images = digits.images.reshape((n, -1))


class NeuralNetwork:
    def __init__(self, input_matrix: "Array"):
        self.input_matrix = input_matrix
        self.biases = None
        self.weights = None
        self.bias_weights = None
        self.output = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.threshold = 0.5
        self.results = list()
        self.bias = np.array([[0.1 for _ in range(len(images[0]))]
                              for _ in range(len(images))])

    def generate_weights(self, num: tuple):
        biases: list[list] = list()
        weights: list[list] = list()
        bias_weights: list[list] = list()

        # Calculating the length of the input (pixels)
        flayer = len(self.input_matrix)

        # Creating the inputs weights by list compreension
        we = bwe = [0.5 for _ in range(flayer)]
        bias = [0.1 for _ in range(flayer)]
        biases.append(bias)
        weights.append(we)
        bias_weights.append(bwe)

        # Loops to create the weights of the hidden layers
        if num.isinstance(tuple):
            for x in num:
                we = bwe = [0.5 for _ in range(x)]
                bias = [0.1 for _ in range(x)]
                biases.append(bias)
                weights.append(we)
                bias_weights.append(bwe)
        else:
            we = bwe = [0.5 for _ in range(num)]
            bias = [0.1 for _ in range(num)]
            biases.append(bias)
            weights.append(we)
            bias_weights.append(bwe)

        # Creating the output weights by list compreension
        we = bwe = [0.5 for _ in self.output]
        bias = [0.1 for _ in self.output]
        biases.append(bias)
        weights.append(we)
        bias_weights.append(bwe)

        # Create a numpy array with the weights
        self.biases = np.array(biases)
        self.weights = np.array(weights)
        self.bias_weights = np.array(bias_weights)

    @staticmethod
    # Method that do the basic perceptron operation
    def perceptron(x, w, b, web):
        return np.multiply(x, w) + np.multiply(b, web)

    @staticmethod
    def activation_function(value: int) -> int or float:
        val = -value
        # Sigmoid function
        return 1 / (1 + exp(val))

    # Method that do the operations between the hidden layers
    def layers(self, layers: tuple | int):
        network_result: list[list] = list()
        for inp in self.input_matrix:
            for index, neurons in enumerate(layers):
                layer_result = list()
                for neuron in range(neurons):
                    result = self.perceptron(inp,
                                             self.weights[index],
                                             self.bias[index],
                                             self.bias_weights[index])
                    sigmoid = self.activation_function(result)

                    if sigmoid > self.threshold:
                        layer_result.append(sigmoid)
                    else:
                        layer_result.append(0)
            for output in self.output:
                

                network_result.append(layer_result)

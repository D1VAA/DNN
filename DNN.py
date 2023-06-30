from typing import Any

import numpy as np
import pandas as pd
from math import exp
from random import shuffle


class NeuralNetwork:
    def __init__(self):
        self.hidden_layers = None
        self.training_data = None
        self.targets = None
        self.biases = None
        self.weights = None
        self.bias_weights = None
        self.output = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.threshold = 0.5
        self.results = list()

    def __generate_weights(self, range_w: list[int, list, int]):
        """
        :param range_w: expect [len(training_data), [number of neurons in each hd layers separated by comma],
        len(output)]
        """
        biases: list[list] = list()
        weights: list[list] = list()
        bias_weights: list[list] = list()
        mean = 0
        std_dev = 1

        for index, r in enumerate(range_w):
            if index == 1:
                # Generate the biases and weights based on the length of the layer
                for ind, neuron in enumerate(r):
                    for i in range(neuron):
                        nin = range_w[index - 1] if ind == 0 else neuron
                        we = bwe = np.random.randn(nin) / np.sqrt(nin)
                        bias = np.random.normal(mean, std_dev, nin)

                        weight_list = we.tolist()
                        bias_list = bias.tolist()
                        weight_bias_list = bwe.tolist()

                        weights.append(weight_list)
                        biases.append(bias_list)
                        bias_weights.append(weight_bias_list)
            elif index == 2:
                for _ in range(range_w[1][-1]):
                    we = bwe = np.random.randn(r) / np.sqrt(r)
                    bias = np.random.normal(mean, std_dev, r)

                    weight_list = we.tolist()
                    bias_list = bias.tolist()
                    weight_bias_list = bwe.tolist()

                    weights.append(weight_list)
                    biases.append(bias_list)
                    bias_weights.append(weight_bias_list)

        # Create a numpy array with the weights
        self.biases = biases
        self.weights = weights
        print(self.weights[-1])
        self.bias_weights = bias_weights

    @staticmethod
    # Method that do the basic perceptron operation
    def __perceptron(x, w, b, web):
        data = np.array(x)
        weight = np.array(w)
        bias = np.array(b)
        bias_weight = np.array(web)
        return np.sum(np.multiply(data, weight) + np.multiply(bias, bias_weight))

    @staticmethod
    def activation_function(value: int) -> int or float:
        return 1 / (1 + exp(-value))  # Sigmoid function

    # Method that do the operations between the hidden layers
    def training_config(self, training_data, target, neurons_layers: list):
        self.training_data = training_data  # Array containg the input of each layer
        self.hidden_layers: list[int] = neurons_layers
        self.targets = target

    def train(self):
        # Section to generate the weights to all layers
        inplayer = len(self.training_data[0])  # Get the lenght of the first item in the training array
        outlayer = len(self.output)
        w_layers = [inplayer, self.hidden_layers, outlayer]
        self.__generate_weights(w_layers)  # Call method that generate the biases and weights
        self.hidden_layers.append(len(self.output))

        for index, data in enumerate(self.training_data):
            self.__layers(data.tolist(), self.hidden_layers, self.targets[index])

    def __layers(self, inp_data, hd_layers: list | int, targets):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        input_matrix.append(inp_data)
        neuron_count = 0
        # print('Número: ', targets)

        for index, neurons in enumerate(hd_layers):
            layer_result = list()  # List containing the neurons output
            print("====> Layer", index + 1)
            print("Weight", len(self.weights[2]))
            for i in range(neurons):
                print(f'Input Matrix {neuron_count}: ', input_matrix[index])
                result = self.__perceptron(input_matrix[index],
                                           self.weights[neuron_count],
                                           self.biases[neuron_count],
                                           self.bias_weights[neuron_count])  # Call the perceptron method (Neuron)
                sigmoid = self.activation_function(result)  # Call the sigmoid activation function

                # The sigmoid return must be greater than the threshold to pass the information to the next neuron
                if sigmoid > self.threshold:
                    layer_result.append(sigmoid)
                else:
                    layer_result.append(0)

                neuron_count += 1
            # Append the layer results at the input matrix
            input_matrix.append(layer_result)
        # print('Resposta: ', input_matrix[-1])

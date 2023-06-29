from typing import Any

import numpy as np
import pandas as pd
from math import exp


class NeuralNetwork:
    def __init__(self):
        self.hidden_layers = None
        self.training_data = None
        self.biases = None
        self.weights = None
        self.bias_weights = None
        self.output = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.threshold = 0.5
        self.results = list()

    def __generate_weights(self, range_w: list[int, list, int], values=None):
        """
        :param range_w: expect [len(input_matrix), [number of neurons in each hd layers separated by comma],
        len(output)]
        :param values: expect a dictionary that identify the value that will be atributted to the biases
        and the weights default: {'weights': 0.5, 'biases': 0.1}
        """
        if values is None:
            values = {'weights': 0.5, 'biases': 0.1}
        biases: list[list] = list()
        weights: list[list] = list()
        bias_weights: list[list] = list()

        for index, r in enumerate(range_w):
            if index == 1:
                # Generate the biases and weights based on the length of the layer
                for neuron in r:
                    we = bwe = [values['weights'] for _ in range(neuron)]
                    bias = [values['biases'] for _ in range(neuron)]
                    weights.append(we)
                    biases.append(bias)
                    bias_weights.append(bwe)
            else:
                we = bwe = [values['weights'] for _ in range(r)]
                bias = [values['biases'] for _ in range(r)]
                weights.append(we)
                biases.append(bias)
                bias_weights.append(bwe)

        # Create a numpy array with the weights
        self.biases = biases
        self.weights = weights
        self.bias_weights = bias_weights

    @staticmethod
    # Method that do the basic perceptron operation
    def __perceptron(x, w, b, web):
        data = np.array(x)
        weight = np.array(w)
        bias = np.array(b)
        bias_weight = np.array(web)
        print(np.sum(np.multiply(data, weight) + np.multiply(bias, bias_weight)))
        return np.sum(np.multiply(data, weight) + np.multiply(bias, bias_weight))

    @staticmethod
    def activation_function(value: int) -> int or float:
        val = -value
        return 1 / (1 + exp(val))  # Sigmoid function

    # Method that do the operations between the hidden layers
    def training_config(self, training_data, neurons_layers: list):
        self.training_data = training_data  # Array containg the input of each layer
        self.hidden_layers: list[int] = neurons_layers

    def train(self):
        # Section to generate the weights to all layers
        inplayer = len(self.training_data[0])  # Get the lenght of the first item in the training array
        outlayer = len(self.output)
        w_layers = [inplayer, self.hidden_layers, outlayer]
        self.__generate_weights(w_layers)  # Call method that generate the biases and weights
        self.hidden_layers.append(len(self.output))

        for data in self.training_data:
            self.__layers(data, self.hidden_layers)

    def __layers(self, inp_data, hd_layers: list | int):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        input_matrix.append(inp_data)

        for index, neurons in enumerate(hd_layers):
            layer_result = list()  # List containing the neurons output
            # print('====> Layer: ', index+1)
            for i in range(neurons):
                #print('>>>> Neuron', i, end='\n\n')
                result = self.__perceptron(input_matrix[index],
                                           self.weights[index],
                                           self.biases[index],
                                           self.bias_weights[index])  # Call the perceptron method (Neuron)
                print(result)
                sigmoid = self.activation_function(result)  # Call the sigmoid activation function

                # The sigmoid return must be greater than the threshold to pass the information to the next neuron
                if sigmoid > self.threshold:
                    layer_result.append(sigmoid)
                else:
                    layer_result.append(0)

            # Append the layer results at the input matrix
            # print(len(layer_result))
            #print('Layer: ', index, layer_result)
            input_matrix.append(layer_result)

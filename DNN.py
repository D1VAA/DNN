import numpy as np
from time import sleep
from bcolors import Colors
from random import shuffle
import pdb


class NeuralNetwork:
    def __init__(self):
        self.layers = None
        self.training_data = None
        self.targets = None
        self.biases = None
        self.weights = None
        self.output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threshold = 2.5

    @classmethod
    def _sigmoid(cls, value: int, deriv=False) -> int or float:
        if value < -100: return 0
        if value > 100: return 1
        if deriv:
            # Return the derivative of sigmoid function
            return cls._sigmoid(value) * (1 - cls._sigmoid(value))
        return 1 / (1 + np.exp(-value))  # Sigmoid function

    @classmethod
    def _softmax(cls, x, deriv=False):
        e_x = np.exp(x - np.max(x))
        if deriv:
            return cls._softmax(x) * (1-cls._softmax(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def __mean_squared_error(preds, targets):
        mse = (targets - preds) ** 2
        deriv_mse = -2 * (targets - preds)
        # Return the mse and the derivative of the error
        return mse, deriv_mse

    @staticmethod
    def _cross_entropy_loss(preds, targets, deriv=False):
        epsilon = 1e-11
        preds = np.clip(preds, epsilon, 1. - epsilon)
        if deriv:
            return - (targets / preds - (1 - targets) / (1 - preds))
        else:
            return - np.sum(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

    def __backpropagation(self, deriv_mse, activations, deriv_activ):
        r_activ = list(reversed(activations))[1:]
        r_deriv_activ = list(reversed(deriv_activ))
        r_biases = list(reversed(self.biases))
        r_weights = list(reversed(self.weights))

        for i, (weights, biases) in enumerate(zip(r_weights, r_biases)):
            neu_out_g_list = [np.sum(d_a * deriv_mse) for d_a in r_deriv_activ[i]]
            neu_out_g_list = np.array(neu_out_g_list).reshape(self.biases[-1 - i].shape)
            self.biases[-1 - i] -= neu_out_g_list

            # Calculate weight gradients
            w_grad = np.array([np.array(x) * r_activ[i] for x in neu_out_g_list])

            # Update weights using gradient descent
            self.weights[-1 - i] -= (self.eta * w_grad)

            if i != 0 and i != len(r_weights) - 1:
                propag = np.array([n * w for n, w in zip(neu_out_g_list, weights)])
                new_neu_grad = np.array([np.sum(x * y) for x, y in zip(r_deriv_activ[1 + i], propag.T)])
                neu_out_g_list = new_neu_grad

    # Method that do the operations between the hidden layers
    def network_config(self, training_data, target, neurons_layers: list, epoch=10, eta=0.01):
        self.training_data = np.array(training_data)  # Array containg the input of each layer
        self.epoch = epoch  # Times to execute all the data training
        self.eta = eta  # Learning rate

        # Inserts the length of the input data at the list begining
        neurons_layers.insert(0, len(training_data.tolist()[0]))
        # Inserts the length of the output data at the list end
        neurons_layers.append(len(self.output))

        # Random normal gaussian variables with mean = 0 and standard deviation = 1/Vnin
        self.weights = [np.random.normal(0, 0.01, (y, x)) for x, y in
                        zip(neurons_layers[:-1], neurons_layers[1:])]
        # Random normal gaussian variables with mean = 0 and standard deviation = 1
        self.biases = [np.random.randn(y, 1) for y in neurons_layers[1:]]
        self.layers: list[int] = neurons_layers
        self.targets = target

    def train(self):
        count = 0
        for e in range(self.epoch):
            indices = np.random.permutation(len(self.training_data))
            data = self.training_data[indices]
            targets = self.targets[indices]
            # For each data in all training data
            for data, target in zip(data, targets):
                print(f'Treinamento Nº: {Colors.BG_RED}{count}{Colors.RESET}\n')
                # Pred = output, activ_h: activations, deriv_h = derivative of each activation
                (pred, activ_h, deriv_h) = self._mlp(data, self.layers[1:], target)
                t_list = np.zeros(10)
                t_list[target] = 1
                error = self._cross_entropy_loss(pred, t_list)
                deriv_mse = self._cross_entropy_loss(pred, t_list, deriv=True)
                self.__backpropagation(deriv_mse, activ_h, deriv_h)  # Call the backpropagation method
                count += 1  # To count the executions

    def _mlp(self, inp_data, hd_layers: list | int, target):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        deriv_sigmoid = list()
        input_matrix.append(inp_data)
        deriv_sigmoid.append(inp_data)

        #  Loop for each layer
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            data = [input_matrix[index] for _ in range(len(weights))]
            lp = [self._softmax((np.dot(data[i], weights[i]) + bias[i])[0])
                  for i in range(len(weights))]
            d_sigmoid = [self._softmax((np.dot(data[i], weights[i]) + bias[i])[0], deriv=True)
                         for i in range(len(weights))]

            input_matrix.append(lp)
            deriv_sigmoid.append(d_sigmoid)

        # Select the output layer
        out_array = input_matrix[-1]
        output = out_array.index(max(out_array))

        print(f'{Colors.PURPLE}[N]{Colors.RESET} >>', end=' ')
        for num, out in enumerate(out_array):
            print(f'{Colors.HARD_RED}[{num}]{Colors.RESET}: [{out:.2f}];', end=' ')
        print(f'\n{Colors.YELLOW}[+]{Colors.RESET} Most Rated : {max(out_array):.2f} ', end='\n')
        i = 'C' if output == target else 'W'
        print(f'{Colors.CIAN}[*]{Colors.RESET} Resposta Correta >> {Colors.GREEN}{target}{Colors.RESET}', end='\n')
        print(f'{Colors.RED}[{i}]{Colors.RESET} Resposta RN >> {Colors.HARD_RED}{output}{Colors.RESET}',
              end='\n\n\n\n')

        # Return the output, the activations of each layer and the derivative of each activation
        return out_array, input_matrix, deriv_sigmoid

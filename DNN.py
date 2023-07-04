import numpy as np
from time import sleep
from bcolors import Colors
from random import shuffle


class NeuralNetwork:
    def __init__(self):
        self.layers = None
        self.training_data = None
        self.targets = None
        self.biases = None
        self.weights = None
        self.output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threshold = 0.6

    @classmethod
    def __activation_function(cls, value: int, deriv=False) -> int or float:
        if deriv:
            # Return the derivative of sigmoid function
            return cls.__activation_function(value) * (1 - cls.__activation_function(value))
        return 1 / (1 + np.exp(-value))  # Sigmoid function

    @staticmethod
    def __mean_squared_error(preds, targets) -> int or float:
        mse = (targets - preds) ** 2
        deriv_mse = -2 * (targets - preds)
        # Return the mse and the derivative of the error
        return mse, deriv_mse

    @staticmethod
    def __cross_entropy_loss(preds, target: int, deriv=False):
        n = len(preds)
        y = np.zeros(n)
        y[target] = 1
        epsilon = 1e-10
        preds = np.clip(preds, epsilon, 1 - epsilon)

        if deriv:
            e_deriv = -(y / preds) + ((1 - y) / (1 - preds))
            return e_deriv

        loss = -np.sum(y * np.log(preds)) / n
        return loss

    def __backpropagation(self, deriv_mse, activations, deriv_activ):
        r_activ = list(reversed(activations))[1:]  # Activation list reversed - Length : 3 (include input layer)
        # Derivative activation list reversed - Length : 4 (include input layer)
        r_deriv_activ = list(reversed(deriv_activ))
        r_biases = list(reversed(self.biases))  # Bias list reversed - Length : 3 (exclude input layer)
        r_weights = list(reversed(self.weights))  # Weights list reversed - Length : 3 (only 3 connections)

        for i, (weights, biases) in enumerate(zip(r_weights, r_biases)):
            # Neurons local gradient
            if i == 0:
                neu_out_g_list = [np.sum(d_a * deriv_mse) for d_a in r_deriv_activ[i]]

                # Bias update
                new_bias = np.array([x - y for x, y in zip(self.biases[-1 - i], neu_out_g_list)])
                self.biases[-1 - i] = new_bias

                # Weight update
                w_grad = np.array([np.array(x) * r_activ[i] for x in neu_out_g_list])
                self.weights[-1 - i] -= (self.eta * w_grad)

                # Propag backwards
                propag = np.array([n * w for n, w in zip(neu_out_g_list, weights)])
                new_neu_grad = np.array([np.sum(x * y) for x, y in zip(r_deriv_activ[1 + i], propag.T)])
                neu_out_g_list = new_neu_grad

            elif i == len(r_weights) - 1:
                new_bias = np.array([x - y for x, y in zip(self.biases[-1 - i], neu_out_g_list)])
                self.biases[-1 - i] = new_bias

                w_grad = np.array([np.array(x) * r_activ[i] for x in neu_out_g_list])
                self.weights[-1 - i] -= (self.eta * w_grad)

            elif i != len(r_weights) - 1 or i != 0:
                w_grad = np.array([np.array(x) * r_activ[i] for x in neu_out_g_list])
                self.weights[-1 - i] -= (self.eta * w_grad)
                propag = np.array([n * w for n, w in zip(neu_out_g_list, weights)])
                new_neu_grad = np.array([np.sum(x * y) for x, y in zip(r_deriv_activ[1 + i], propag.T)])
                neu_out_g_list = new_neu_grad

                new_bias = np.array([x - y for x, y in zip(self.biases[-1 - i], neu_out_g_list)])
                self.biases[-1 - i] = new_bias

    # Method that do the operations between the hidden layers
    def network_config(self, training_data, target, neurons_layers: list, epoch=10, eta=0.05):
        self.training_data = np.array(training_data)  # Array containg the input of each layer
        self.epoch = epoch  # Times to execute all the data training
        self.eta = eta  # Learning rate

        # Inserts the length of the input data at the list begining
        neurons_layers.insert(0, len(training_data.tolist()[0]))
        # Inserts the length of the output data at the list end
        neurons_layers.append(len(self.output))

        # Random normal gaussian variables with mean = 0 and standard deviation = 1/Vnin
        self.weights = [np.random.normal(0, 1 / np.sqrt(x), (y, x)) for x, y in
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
                (pred, activ_h, deriv_h) = self.__layers(data, self.layers[1:], target)
                deriv_mse = self.__cross_entropy_loss(pred, target, deriv=True)
                self.__backpropagation(deriv_mse, activ_h, deriv_h)  # Call the backpropagation method
                count += 1  # To count the executions

    def __layers(self, inp_data, hd_layers: list | int, target):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        deriv_sigmoid = list()
        input_matrix.append(inp_data)
        deriv_sigmoid.append(inp_data)

        #  Loop for each layer
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            data = [input_matrix[index] for _ in range(len(weights))]
            lp = [(np.dot(data[i], weights[i]) + bias[i])[0] for i in range(len(weights))]
            r_sigmoid = [self.__activation_function(o) for o in lp]
            d_sigmoid = [self.__activation_function(o, deriv=True) for o in lp]
            l_result = [x if x > self.threshold else 0 for x in r_sigmoid]
            input_matrix.append(l_result)
            deriv_sigmoid.append(d_sigmoid)

        out_array = input_matrix[-1]  # Select the output layer
        # Get the index of the activated neuron at the output layer
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

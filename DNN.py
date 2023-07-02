import numpy as np
from time import sleep
from bcolors import Colors


class NeuralNetwork:
    def __init__(self):
        self.hidden_layers = None
        self.training_data = None
        self.targets = None
        self.biases = None
        self.weights = None
        self.epoch = None
        self.batch_size = None
        self.eta = None
        self.output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threshold = 0.6
        self.results = list()

    @staticmethod
    def __activation_function(value: int, deriv=False) -> int or float:
        if deriv:
            return value * (1 - value)
        return 1 / (1 + np.exp(-value))  # Sigmoid function

    @staticmethod
    def __mean_squared_error(preds, targets) -> int or float:
        length = len(preds)
        preds = np.array(preds)
        targets = np.array(targets)
        squared_error = [(tar - val) ** 2 for tar, val in zip(targets, preds)]
        mse = (np.sum(squared_error) / length)
        return mse

    def __backpropagation(self, entry_data, preds, targets, activations):
        out_error = self.__mean_squared_error(preds, targets)
        for i, activ in enumerate(activations):
            layers_error = list()
            r_activ = activ[::-1]  # Return the list reversed
            out_delta = np.array([out_error * self.__activation_function(y, deriv=True) for y in
                                  r_activ[0]])  # Calculating the gradient of the output layer
            layers_error.append(out_delta)
            for layer, (i, a) in zip(reversed(self.weights), enumerate(r_activ)):
                transpose_weights = layer.T  # Transpose the weights of the current layer

                # How much each weight impacted at the error of each neuron
                print('Layer Error: \n\n', layers_error[-1], '\n\n')
                print('Layer: ', layer)
                hdl_error = np.array([np.dot(d, l).tolist() for d, l in zip(layers_error[-1], layer)])

                # Calculating the derivative of the active function of each activation of the current layer
                deriv_activ = np.array([np.array(self.__activation_function(s, deriv=True)) for s in r_activ[1 + i]])

                # Calculating the gradient of each neuron
                hdl_delta = np.array([x * y for x, y in zip(hdl_error.T, deriv_activ)])
                weight_update = transpose_weights - self.eta * hdl_delta
                # print('Derivative: \n\n', deriv_activ)
                print('Hdl Error: \n\n', hdl_error.T)
                print('HDL Delta: \n\n', hdl_delta)
                # print('Transpose: \n\n', transpose_weights, '\n\n')
                print('Weights: \n\n', weight_update)
                print('Activ', r_activ[1 + i])

                layers_error.append(hdl_delta)
                sleep(2)

    # Method that do the operations between the hidden layers
    def network_config(self, training_data, target, neurons_layers: list, epoch=10, batch_size=10, eta=0.01):
        self.training_data = training_data  # Array containg the input of each layer
        self.epoch = epoch
        self.batch_size = batch_size
        self.eta = eta

        neurons_layers.insert(0, len(training_data.tolist()[0]))
        neurons_layers.append(len(self.output))

        self.weights = [np.random.normal(0, 1 / np.sqrt(x), (y, x)) for x, y in
                        zip(neurons_layers[:-1], neurons_layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in neurons_layers[1:]]
        self.hidden_layers: list[int] = neurons_layers
        self.targets = target

    def train(self):
        # Section to generate the weights to all layers
        preds = list()
        targets = list()
        inp_tests = list()
        activ_h_l = list()
        count = 0
        batches = [self.training_data[batch:batch + self.batch_size] for batch in
                   range(0, len(self.training_data), self.batch_size)]

        for index, data in enumerate(batches):
            for entry in data:
                print(f'Treinamento Nº: {Colors.BG_RED}{count}{Colors.RESET}\n')
                (pred, activ_h) = self.__layers(entry, self.hidden_layers[1:], self.targets[count])
                targets.append(self.targets[count])
                preds.append(pred)
                inp_tests.append(entry.tolist())
                activ_h_l.append(activ_h)
                count += 1
            self.__backpropagation(inp_tests, preds, targets, activ_h_l)

    def __layers(self, inp_data, hd_layers: list | int, target):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        input_matrix.append(inp_data)

        #  Loop for each layer
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            data = [input_matrix[index] for _ in range(len(weights))]
            lp = [(np.dot(data[i], weights[i]) + bias[i])[0] for i in range(len(weights))]
            r_sigmoid = [self.__activation_function(o) for o in lp]
            l_result = [x if x > self.threshold else 0 for x in r_sigmoid]
            input_matrix.append(l_result)

        out_array = input_matrix[-1]
        output = out_array.index(max(out_array))

        #
        print(f'{Colors.PURPLE}[N]{Colors.RESET} >>', end=' ')
        for num, out in enumerate(out_array):
            print(f'{Colors.HARD_RED}[{num}]{Colors.RESET}: [{out:.2f}];', end=' ')
        print(f'\n{Colors.YELLOW}[+]{Colors.RESET} Most Rated : {max(out_array):.2f} ', end='\n')
        i = 'C' if output == target else 'W'
        print(f'{Colors.CIAN}[*]{Colors.RESET} Resposta Correta >> {Colors.GREEN}{target}{Colors.RESET}', end='\n')
        print(f'{Colors.RED}[{i}]{Colors.RESET} Resposta RN >> {Colors.HARD_RED}{output}{Colors.RESET}',
              end='\n\n\n\n')
        return output, input_matrix

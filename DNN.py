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

    @classmethod
    def __activation_function(cls, value: int, deriv=False) -> int or float:
        if deriv:
            return cls.__activation_function(value) * (1 - cls.__activation_function(value))
        return 1 / (1 + np.exp(-value))  # Sigmoid function

    @staticmethod
    def __mean_squared_error(preds, targets) -> int or float:
        mse = (((targets - preds) ** 2) / 2)
        return mse

    def __backpropagation(self, entry_data, preds, targets, activations):
        out_error = self.__mean_squared_error(preds, targets)
        new_weights = list()
        layers_error = list()

        r_activ = activations[::-1]  # Return the list reversed
        out_delta = np.array([out_error * self.__activation_function(y, deriv=True) for y in r_activ[0]])  # Calculating the gradient of the output layer

        layers_error.append(out_delta)

        for layer, (i, a) in zip(reversed(self.weights), enumerate(r_activ)):
            t_weights= layer.T  # Transpose the weights of the current layer

            # How much each weight impacted at the error of each neuron
            hdl_error = np.array([np.dot(d,l).tolist() for d, l in zip(layers_error[-1], layer.T)])
            deriv_activ = np.array([np.array(self.__activation_function(s, deriv=True)) for s in r_activ[1 + i]])
            hdl_delta = np.array([x * y for x, y in zip(hdl_error, deriv_activ)])
            print('Hdl Error: ', hdl_error)
            print(layer)
            print('\n\n',hdl_delta)
            weight_update = layer - (self.eta * hdl_delta)
            print(weight_update)


        return new_weights[::-1]

    # Method that do the operations between the hidden layers
    def network_config(self, training_data, target, neurons_layers: list, epoch=10, eta=5):
        self.training_data = training_data  # Array containg the input of each layer
        self.epoch = epoch
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
        count = 0
        for e in range(self.epoch):
            for data, target in zip(self.training_data, self.targets):
                print(f'Treinamento Nº: {Colors.BG_RED}{count}{Colors.RESET}\n')
                (pred, activ_h) = self.__layers(data, self.hidden_layers[1:], target)
                count += 1
                self.weights = self.__backpropagation(data[0], pred, target, activ_h)

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

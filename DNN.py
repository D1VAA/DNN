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
        self.output = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.threshold = 0.5
        self.results = list()

    @staticmethod
    # Method that do the basic perceptron operation
    def __perceptron(x, w, b):
        data = np.array(x)
        weight = np.array(w)
        bias = np.array(b)
        result = np.sum(np.multiply(data, weight)) + bias
        return result.item()

    @staticmethod
    def __activation_function(value: int) -> int or float:
        return 1 / (1 + np.exp(-value))  # Sigmoid function

    # Method that do the operations between the hidden layers
    def training_config(self, training_data, target, neurons_layers: list):
        self.training_data = training_data  # Array containg the input of each layer

        neurons_layers.insert(0, len(training_data.tolist()[0]))
        neurons_layers.append(len(self.output))

        self.weights = [np.random.randn(y, x) for x, y in zip(neurons_layers[:-1], neurons_layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in neurons_layers]
        self.hidden_layers: list[int] = neurons_layers
        self.targets = target

    def train(self):
        # Section to generate the weights to all layers
        for index, data in enumerate(self.training_data):
            print(f'Treinamento Nº: {Colors.BG_RED}{index}{Colors.RESET}\n')
            self.__layers(data.tolist(), self.hidden_layers[1:], self.targets[index])

    def __layers(self, inp_data, hd_layers: list | int, target):
        # Array that will contain all the inputs of each layer
        input_matrix = list()
        input_matrix.append(inp_data)
        neuron_count = 0

        for index, neurons in enumerate(hd_layers):
            layer_result = list()  # List containing the neurons output
            for i in range(neurons):
                result = self.__perceptron(input_matrix[index],
                                           self.weights[index][i],
                                           self.biases[index][i])  # Call the perceptron (Neuron)
                sigmoid = self.__activation_function(result)  # Call the sigmoid activation function

                # The sigmoid returns must be greater than the threshold to activate
                if sigmoid > self.threshold:
                    layer_result.append(sigmoid)
                else:
                    layer_result.append(0)

                neuron_count += 1
            # Append the layer results at the input matrix
            input_matrix.append(layer_result)
        out_array = input_matrix[-1]
        output = out_array.index(max(out_array))

        print(f'{Colors.CIAN}[*]{Colors.RESET} Resposta Correta >> {Colors.GREEN}{target}{Colors.RESET}',
              end='\n')

        print(f'{Colors.PURPLE}[N]{Colors.RESET} >>', end=' ')
        for num, out in enumerate(out_array):
            print(f'{Colors.HARD_RED}[{num}]{Colors.RESET}: [{out:.2f}];', end=' ')
        print(f'\n{Colors.YELLOW}[+]{Colors.RESET} Most Rated : {max(input_matrix[-1]):.2f} ', end='\n')
        i = 'C' if output == target else 'W'
        print(f'{Colors.RED}[{i}]{Colors.RESET} Resposta RN >> {Colors.HARD_RED}{output}{Colors.RESET}',
              end='\n\n\n\n')

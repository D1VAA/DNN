import numpy as np
from packages.bcolors import Colors
from NeuralNetwork.activations_f import sigmoid, softmax
from NeuralNetwork.cost_functions import cross_entropy_loss, mean_squared_error
from NeuralNetwork.bpropag import backpropagation


class MultiLayerPerceptron:
    def __init__(self,
                 training_data,
                 target,
                 n_layers: list[int],
                 epoch=50,
                 eta=0.01):

        # Set initial parameters
        self.training_data = np.array(training_data)
        self.targets = np.array(target)
        self.threshold = 2.5
        self.epoch = epoch
        self.eta = eta
        self.soft = None
        n_layers.insert(0, len(training_data[0]))
        n_layers.append(10)
        self.layers = n_layers

        # Random normal gaussian variables with mean = 0 and standard deviation = 1
        self.weights = [np.random.normal(0, 0.01, (y, x)) for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]

    def train(self):
        # Run all epochs
        for _ in range(self.epoch):
            indices = np.random.permutation(len(self.training_data))
            data = self.training_data[indices]
            targets = self.targets[indices]

            for data, target in zip(data, targets):
                pred, activ_h, deriv_h = self._mlp(data, target)
                t_list = np.zeros(10)
                t_list[target] = 1
                cel, d_cel = cross_entropy_loss(pred, t_list)
                backpropagation(d_cel, activ_h, deriv_h)

    def _mlp(self, inp_data, target):
        activations = [inp_data]
        deriv_sigmoid = [inp_data]

        # Loop for each layer
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            # lp = sigmoid(perceptron)
            lp = [sigmoid((np.dot(activations[-1], weights[i]) + bias[i])[0]) for i in range(len(weights))]

            if index != len(self.weights) - 1:
                deriv_sigmoid.append([x['d_sig'] for x in lp])
            # Execute the softmax function on the output layer
            else:
                self.soft = softmax([x['val'] for x in lp])
            activations.append([x['sig'] for x in lp])

        # Select the output layer
        out_layer = list(self.soft['soft'])
        output = out_layer.index(max(out_layer))
        i = 'C' if output == target else 'W'

        print(f'{Colors.PURPLE}[N]{Colors.RESET} >>', end=' ')
        for num, out in enumerate(out_layer):
            print(f'{Colors.HARD_RED}[{num}]{Colors.RESET}: [{out:.2f}];', end=' ')

        print(f'\n{Colors.YELLOW}[+]{Colors.RESET} Most Rated : {max(out_layer):.2f} ', end='\n')
        print(f'{Colors.CIAN}[*]{Colors.RESET} Resposta Correta >> {Colors.GREEN}{target}{Colors.RESET}', end='\n')
        print(f'{Colors.RED}[{i}]{Colors.RESET} Resposta RN >> {Colors.HARD_RED}{output}{Colors.RESET}')
        print('\n\n\n')

        # Return the output, the activations of each layer and the derivative of each activation
        return out_layer, activations, deriv_sigmoid

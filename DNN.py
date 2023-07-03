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
        # mse = np.square(np.subtract(targets, preds)).mean()
        mse = (targets - preds) ** 2
        deriv_mse = -2 * (targets - preds)
        return mse, deriv_mse

    def __backpropagation(self, preds, targets, activations, deriv_activ):
        # Retorna o erro quadrático médio e a derivada do mesmo
        out_error, deriv_out_error = self.__mean_squared_error(preds, targets)

        # listas para organizar os resultados
        layers_error = list()
        c = 1
        new_weights = list()
        new_bias = list()

        # Retorna a lista que contém as ativações de cada layer de forma invertida (da saída para a entrada)
        r_activ = activations[::-1]
        # O mesmo que a lista anterior, mas com as derivadas das funções de ativação
        r_deriv_activ = deriv_activ[::-1]
        r_biases = self.biases[::-1]  # Lista com os bias de cada camada invertida
        weights = self.weights[::-1]  # Lista com os pesos mas invertida

        # Get the position of the neuron that were activated on the output layer
        i_actived_neuron = r_activ[0].index(max(r_activ[0]))

        # Calc the gradient of the neuron activated in the output layer
        out_grad = np.array(deriv_out_error * r_deriv_activ[0][i_actived_neuron])  # Derivative of e by z
        f_up = out_grad * r_activ[1]
        weight_update = self.weights[-1][i_actived_neuron] - (self.eta * f_up)
        weights[0][i_actived_neuron] = weight_update

        propag = np.array(f_up * weights[0][i_actived_neuron]) * r_deriv_activ[1]
        layers_error.append(propag)

        for i, layer in enumerate(weights[1:]):
            w_grad = [x * r_activ[2 + i] for x in layers_error[-1]]
            w_up = layer - (self.eta * w_grad)
            weights[i + 1] = w_up

            # Updating variables
            propag = np.dot(r_deriv_activ[1 + i], np.array(layers_error[-1] * layer))
            print(propag)
            layers_error.append(propag)
            sleep(2)

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
                (pred, activ_h, deriv_h) = self.__layers(data, self.hidden_layers[1:], target)
                count += 1
                self.weights = self.__backpropagation(pred, target, activ_h, deriv_h)

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
        return output, input_matrix, deriv_sigmoid

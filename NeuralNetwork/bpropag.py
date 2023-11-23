import numpy as np


def backpropagation(self, deriv_mse, activations, deriv_activ):
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

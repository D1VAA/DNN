import numpy as np


def sigmoid(value: int, deriv=True) -> ["input", "sigmoid", "derivative of sigmoid"]:
    """
    :returns: value and sigmoid or derivative of sigmoid
    :param value: the perceptron value
    """
    if value < -100: return 0
    if value > 100: return 1
    sig = 1 / (1 + np.exp(-value))
    deriv_sig = sig * (1 - sig)
    return {'val': value, 'sig': sig, 'd_sig': deriv_sig}


def softmax(x):
    e_x = np.exp(x - np.max(x))  # improve the numerical stability
    s = e_x / e_x.sum(axis=0)  # calculate the softmax first
    soft = e_x / e_x.sum(axis=0)
    deriv_softmax = np.diagflat(s) - np.dot(s, s.T)
    return {'soft': soft, 'd_soft': deriv_softmax}

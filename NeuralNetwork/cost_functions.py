import numpy as np


def mean_squared_error(preds, targets):
    mse = (targets - preds) ** 2
    deriv_mse = -2 * (targets - preds)
    # Return the mse and the derivative of the error
    return mse, deriv_mse


def cross_entropy_loss(preds, targets):
    """
    returns: cross_entropy_loss, derivative of cross_entropy_loss
    """
    epsilon = 1e-11
    preds = np.clip(preds, epsilon, 1. - epsilon)
    cel = - np.sum(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
    d_cel = - (targets / preds - (1 - targets) / (1 - preds))
    return cel, d_cel
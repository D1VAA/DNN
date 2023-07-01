import numpy as np


class SGD:
    @staticmethod
    def mean_squared_error(preds, targets):
        length = len(preds)
        preds = np.array(preds)
        targets = np.array(targets)
        squared_error = [(tar - val) ** 2 for tar, val in zip(targets, preds)]
        mse = (np.sum(squared_error) / length)
        self.backpropagation()
        return mse

    @staticmethod

    def backpropagation():

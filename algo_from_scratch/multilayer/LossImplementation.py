import numpy as np

from AbstractClasses import Loss


# class CrossEntropyLoss(Loss):
#     def __init__(self):
#         super().__init__("CrossEntropyLoss")
#
#     def calculate_loss(self, y_true, y_predicted):
#         epsilon = 1e-15
#         y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
#         return -np.sum(np.sum(y_true * np.log(y_predicted), axis=0))

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__("CrossEntropyLoss")

    def calculate_loss(self, y_true, y_predicted):
        # Adding epsilon to avoid log(0)
        epsilon = 1e-15
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_predicted), axis=1)
        return np.mean(loss)

    def backward(self, y_true, y_predicted):
        # Adding epsilon to avoid division by zero
        epsilon = 1e-15
        y_predicted = np.clip(y_predicted, epsilon, 1 - epsilon)
        batch_size = y_true.shape[0]
        return (y_predicted - y_true) / batch_size

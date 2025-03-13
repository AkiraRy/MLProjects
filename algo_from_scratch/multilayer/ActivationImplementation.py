import numpy as np
from AbstractClasses import Activation


class ReLU(Activation):
    def __init__(self):
        self.input_x = None

    def __str__(self):
        return 'ReLU'

    def forward(self, x):
        self.input_x = x
        return np.maximum(0, x)

    def backward(self, dO):
        """
        Input:
        - dO: derivative output of previous layer

        returns
        - dx: Gradient with respect to x
        """
        dx = np.copy(self.input_x)
        dx[dx > 0] = 1
        dx[dx <= 0] = 0
        dx = dO * dx # dx is a matrice of 1 and 0
        return dx


class SoftMax(Activation):
    def __init__(self):
        self.output = None

    def __str__(self):
        return 'SoftMax'

    def forward(self, x):
        x -= np.max(x, axis=-1, keepdims=True)
        exponents = np.exp(x)
        output = exponents / np.sum(exponents, axis=-1, keepdims=True)
        self.output = output
        return output

    # def backward(self, dO, paired_with_cross_entropy=False):
    #     # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    #
    #     if paired_with_cross_entropy:
    #         return dO
    #
    #     # Assuming self.output shape is (batch_size, num_classes)
    #     batch_size, num_classes = self.output.shape
    #
    #     del_u = np.zeros_like(self.output)
    #
    #     for i in range(batch_size):
    #         y = self.output[i].reshape(-1, 1)
    #         jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
    #         del_u[i] = np.dot(jacobian_matrix, dO[i])
    #
    #     return del_u

    def backward(self, dO, paired_with_cross_entropy=False):
        if paired_with_cross_entropy:
            # If paired with cross-entropy, del_v is already the simplified gradient
            return dO
        size = np.size(self.output)
        matrix = np.tile(self.output, size)
        return np.dot(
            matrix * (np.identity(size) - np.transpose(matrix)),
            dO
        )


if __name__ == '__main__':
    softmax_layer = SoftMax()  # Assume SoftMax is a defined class
    softmax_layer.output = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])  # Example output (batch_size=2, num_classes=3)
    dO = np.array([[1.0, 2.0, 3.0], [0.5, 0.2, 0.3]])  # Example gradient

    # Compute the backward pass
    grad_input = softmax_layer.backward(dO)
    print(grad_input)

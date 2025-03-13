from AbstractClasses import Optimizer


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr, 'Stochastic Gradient Descent')

    def apply_gradients(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.lr * grad

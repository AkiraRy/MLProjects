import numpy as np


def loss_function(data,theta):
    #get m and b
    m = theta[0]
    b = theta[1]
    loss = 0
    #on each data point
    for i in range(0, len(data)):
        #get x and y
        x = data[i, 0]
        y = data[i, 1]
        #predict the value of y
        y_hat = (m*x + b)
        #compute loss as given in quation (2)
        loss = loss + ((y - (y_hat)) ** 2)
    #mean sqaured loss
    mean_squared_loss = loss / float(len(data))
    return mean_squared_loss

def compute_gradients(data, theta):
    gradients = np.zeros(2)
    #total number of data points
    N = float(len(data))
    m = theta[0]
    b = theta[1]
    #for each data point
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        #gradient of loss function with respect to m as given in (3)
        gradients[0] += - (2 / N) * x * (y - (( m* x) + b))
        #gradient of loss funcction with respect to b as given in (4)
        gradients[1] += - (2 / N) * (y - ((theta[0] * x) + b))
    #add epsilon to avoid division by zero error
    epsilon = 1e-6
    gradients = np.divide(gradients, N + epsilon)
    return gradients


def Momentum(data, theta, lr = 1e-2, gamma = 0.9, num_iterations = 5000):
    loss = []
    #Initialize vt with zeros:
    vt = np.zeros(theta.shape[0])
    for t in range(num_iterations):
        #compute gradients with respect to theta
        gradients = compute_gradients(data, theta)
        #Update vt by equation (8)
        vt = gamma * vt + lr * gradients
        #update model parameter theta by equation (9)
        theta = theta - vt
        #store loss of every iteration
        loss.append(loss_function(data,theta))
    return loss


initial_theta = np.array([1.0, 2.0, 3.0])
optimized_theta = Momentum(theta=initial_theta)

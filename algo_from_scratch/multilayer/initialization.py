import numpy as np


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-s, high=s, size=shape)

#
# from PIL import Image
#
# # Load the image (grayscale)
# image = Image.open('dataset/A/A1.png').convert('L')
#
# # Resize the image to 100x100 pixels
# resized_image = image.resize((28, 28), Image.LANCZOS)
#
# # Save the resized image
# image.show()
# resized_image.show()


# import string
# Convert categorical labels to one-hot encoded vectors


# # Example labels
# y_labels = list(string.ascii_uppercase)
#
# softmax = SoftMax()
# one_hot_labels = labels_to_one_hot(y_labels)
# softmax_probabilities_values = np.random.rand(26,1)
# softmax_probabilites = softmax.forward(softmax_probabilities_values)
#
#
# def cross_entropy_loss(y_true, y_predicted):
#     return np.sum(-1 * np.sum(y_true * np.log(y_predicted), axis=0))
#
#
# for i in range(26):
#     loss = cross_entropy_loss(one_hot_labels[i], np.transpose(softmax_probabilites))
#     # Compute the loss
#     print("Cross-entropy loss:", loss)

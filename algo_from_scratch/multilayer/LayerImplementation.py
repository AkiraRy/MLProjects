from typing import Callable
from AbstractClasses import Layer
import numpy as np
from ActivationImplementation import ReLU, SoftMax
from initialization import glorot_uniform

## https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf


class ConvolutionalLayer(Layer):
    def backward(self, do):
        batch_size, height, width, grads_depth = do.shape

        full_dim_for_grads = (height - 1) * self.stride + 1

        # bias
        if self.use_bias:
            del_b = np.sum(do, axis=(0, 1, 2)) / batch_size

        # for input
        input_dim_without_pad = self.input_x.shape[1] - 2 * self.padding
        input_depth = self.input_x.shape[3]
        del_input = np.zeros((batch_size, input_dim_without_pad, input_dim_without_pad, input_depth))

        del_w = np.zeros((self.num_filters, self.kernel_size, self.kernel_size, input_depth))

        # dilated gradient
        del_o_dil = np.zeros((batch_size, full_dim_for_grads, full_dim_for_grads, self.num_filters))
        del_o_dil[:, :: self.stride, :: self.stride, :] = do

        for filter_index in range(self.num_filters):
            for height in range(self.kernel_size):
                for width in range(self.kernel_size):
                    height_to = height + full_dim_for_grads
                    width_to = width + full_dim_for_grads
                    input_slice = self.input_x[:, height :height_to, width: width_to, :]
                    input_convolved_with_grads = np.sum(input_slice * np.reshape(
                                            del_o_dil[:, :, :, filter_index], del_o_dil.shape[: 3] + (1,)),
                                                                            axis=(1, 2))
                    mean_batched = np.mean(input_convolved_with_grads, axis=0)

                    del_w[filter_index, height, width, :] = mean_batched

        # we rotate this so it will align with the do matrix
        weights_rotated = np.rot90(np.transpose(self.weights, (3, 1, 2, 0)), 2, axes=(1, 2))
        del_o_dil_pad = np.pad(del_o_dil, (
            (0,), (self.kernel_size - 1 - self.padding,), (self.kernel_size - 1 - self.padding,), (0,)),
                                  mode='constant')

        for batch_index in range(batch_size):
            for filter_index in range(input_depth):
                for height in range(input_dim_without_pad):
                    for width in range(input_dim_without_pad):
                        height_to = height + self.kernel_size
                        width_to = width + self.kernel_size
                        gradient_slice = del_o_dil_pad[batch_index, height:height_to, width: width_to, :]
                        weights_convolved_grads = np.sum(gradient_slice * weights_rotated[filter_index])
                        del_input[batch_index, height, width, filter_index] = weights_convolved_grads

        return del_input

    def __str__(self):
        return f'Conv(filter={self.num_filters}, kernel={self.kernel_size}, stride={self.stride}, padding={self.padding})'

    def __init__(self,
                 kernel_size: int = 3,
                 num_filters: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 use_bias: bool = False,
                 initializer: Callable = glorot_uniform):

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.initializer = initializer
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.bias = None
        self.weights = None
        self.weights_setup = False
        self.input_x = None

    def forward(self, Input):
        batch_number, height, width, input_depth = Input.shape
        output_dim_temp = height - self.kernel_size + 2 * self.padding
        assert output_dim_temp % self.stride == 0, (f"In the convolutional layer, with stride {self.stride}"
                                                    f" can`t process whole image evenly.")

        output_dim = output_dim_temp // self.stride + 1
        output_depth = self.num_filters

        if not self.weights_setup:
            self.setup_weights(input_depth)
            self.weights_setup = True

        padded_input = np.pad(Input, ((0,), (self.padding,), (self.padding,), (0,)), mode='constant')
        # shape is batch_number, y, x, depth

        new_shape = (batch_number, output_dim, output_dim, output_depth)
        output_matrix = np.zeros(new_shape)

        for batch_index in range(batch_number):
            for height in range(output_dim):
                for width in range(output_dim):
                    height_from = height * self.stride
                    height_to = height_from + self.kernel_size
                    width_from = width * self.stride
                    width_to = width_from + self.kernel_size

                    image_slice = padded_input[batch_index, height_from:height_to, width_from: width_to, :]

                    for filter_index in range(output_depth):
                        output_matrix[batch_index, height, width, filter_index] = \
                            np.sum(image_slice * self.weights[filter_index]) + (self.bias[filter_index] if self.use_bias else 0)

        self.input_x = np.copy(padded_input)

        return output_matrix

    def setup_weights(self, depth):
        # print((self.num_filters, self.kernel_size, self.kernel_size, depth))
        self.weights = self.initializer(shape=(self.num_filters, self.kernel_size, self.kernel_size, depth))

        if self.use_bias:
            self.bias = np.random.randn(self.num_filters)
        else:
            self.bias = np.zeros(self.num_filters)


class PollingLayer(Layer):
    def __str__(self):
        return f'Polling(kernel={self.kernel_size}, stride={self.stride}, mode={self.mode})'

    def __init__(self, kernel_size: int, stride: int, mode="max_pool"):
        self.index_matrix = None
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def average_pool(self, Input):
        pass

    def max_pool(self, Input: np.ndarray):
        batch_number, height, width, input_depth = Input.shape
        output_dim_temp = height - self.kernel_size

        assert output_dim_temp % self.stride == 0, (f"In the convolutional layer, with stride {self.stride}"
                                                    f" can`t process whole image evenly.")

        output_dim = output_dim_temp // self.stride + 1

        output_matrix = np.zeros((batch_number, output_dim, output_dim, input_depth))
        self.index_matrix = np.zeros((batch_number, output_dim, output_dim, input_depth)).astype(np.int32)
        self.input_shape = Input.shape

        for batch_index in range(batch_number):
            for height in range(output_dim):
                for width in range(output_dim):
                    height_from = height * self.stride
                    height_to = height_from + self.kernel_size
                    width_from = width * self.stride
                    width_to = width_from + self.kernel_size

                    image_slice = Input[batch_index, height_from:height_to, width_from: width_to, :]
                    for d in range(input_depth):
                        output_matrix[batch_index, height, width, d] = np.max(image_slice[:, :, d])
                        self.index_matrix[batch_index, height, width, d] = np.argmax(image_slice[:, :, d])

        return output_matrix

    def forward(self, Input):
        if self.mode == "max_pool":
            return self.max_pool(Input)
        return self.average_pool(Input)

    def backward(self, do):
        if self.mode == "max_pool":
            return self.max_pool_d(do)
        return self.average_pool_d(do)

    def max_pool_d(self, do):
        del_input = np.zeros(self.input_shape)
        batch_size, height, width, depth = do.shape

        for batch_index in range(batch_size):
            for h in range(height):
                for w in range(width):
                    height_from = h * self.stride
                    width_from = w * self.stride

                    for d in range(depth):
                        max_index = self.index_matrix[batch_index, h, w, d]
                        max_height_index = max_index // self.kernel_size
                        max_width_index = max_index % self.kernel_size

                        del_input[batch_index, height_from + max_height_index, width_from + max_width_index, d] += do[
                            batch_index, h, w, d]

        return del_input


    def average_pool_d(self, do):
        pass


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def __str__(self):
        return 'Flatten'

    def forward(self, Input):
        self.input_shape = Input.shape
        output_matrix = np.reshape(Input, (self.input_shape[0], np.prod(self.input_shape[1:])))
        return output_matrix
        # return np.transpose(output_matrix)

    def backward(self, do):
        d_x = np.reshape(do, self.input_shape)
        return d_x


class Dense(Layer):
    def __init__(self, output_dim, initializer: Callable = glorot_uniform, use_bias: bool = True):
        self.d_bias = None
        self.d_weights = None
        self.initializer = initializer
        self.use_bias = use_bias
        self.output_dim = output_dim
        self.weights_setup = False

        # trainable params
        self.weights = None
        self.bias = None

        # data
        self.input_x = None
        self.output = None

    def setup_weights(self, shape):
        self.weights = self.initializer(shape)
        if self.use_bias:
            self.bias = np.zeros(self.output_dim, )
        else:
            self.bias = None

    def forward(self, Input):
        # print(f"{Input.shape = }")
        if not self.weights_setup:
            # input_dim, output_dim
            self.setup_weights((Input.shape[1], self.output_dim))
            self.weights_setup = True

        self.input_x = Input

        self.output = np.dot(Input, self.weights)
        if self.use_bias:
            self.output += self.bias
        return self.output

    def backward(self, do):
        batch_size = 1
        if len(do.shape) > 1:
            batch_size = do.shape[0]
        self.d_weights = np.dot(self.input_x.T, do)
        if self.use_bias:
            self.d_bias = np.sum(do, axis=0, keepdims=True)
        d_input = np.dot(do, self.weights.T)
        return d_input

    def __str__(self):
        return f'Dense(output_dim={self.output_dim})'


def main():
    # Print the shape of the batch matrix
    # print("Shape of batch matrix:", batch_matrix.shape)

    # # Example usage
    # z, y, x = 5, 4, 3  # dimensions of the array
    # #
    # # array = glorot_uniform((z, y, x))
    # array = np.random.rand(2, z, y, x)
    # print(array)
    # print(array.shape)

    array = np.array([[1, 0, 0.5, 0.5],
                      [0, 0.5, 1, 0],
                      [0, 1, 0.5, 1],
                      [1, 0.5, 0.5, 1]])
    print(array.shape)
    convo_layer = ConvolutionalLayer(kernel_size=(2, 2), num_filters=1, stride=1, padding=1, use_bias=False)
    print(convo_layer.filters)
    forward_pass = convo_layer.forward(array)[0]
    print(forward_pass[0])
    print(forward_pass[0].shape)


def main_test_initializer():
    num_filters = 32  # Number of filters (output channels)
    kernel_size = 3  # Kernel size (e.g., 3x3 kernel)
    num_channels = 3  # Number of input channels (e.g., RGB channels)

    # Constructing the shape tuple
    example_shape = (num_filters, kernel_size, kernel_size, num_channels)
    example_filter_weights = glorot_uniform(shape=example_shape)

    print(example_filter_weights[0])
    print(example_filter_weights[0].shape)


def test():
    np.random.seed(42)

    input_data = np.random.randn(28, 28)  # Example 28x28 input image
    input_data = input_data.reshape(1, 28, 28, 1)
    # print(input_data.shape)
    conv_layer = ConvolutionalLayer(num_filters=32, kernel_size=3, stride=1, padding=1)
    output_data = conv_layer.forward(input_data)[0]
    # print(output_data[0])

    print("Output shape:", output_data.shape)


def test_forward():
    array_string = """[[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3, 18, 18, 18,126,136,
      175, 26,166,255,247,127,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0, 30, 36, 94,154,170,253,253,253,253,253,
      225,172,253,242,195, 64,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0, 49,238,253,253,253,253,253,253,253,253,251,
       93, 82, 82, 56, 39,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0, 18,219,253,253,253,253,253,198,182,247,241,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0, 80,156,107,253,253,205, 11,  0, 43,154,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,  1,154,253, 90,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,139,253,190,  2,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 11,190,253, 70,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 35,241,225,160,108,  1,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 81,240,253,253,119,
       25,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 45,186,253,253,
      150, 27,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 93,252,
      253,187,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,249,
      253,249, 64,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 46,130,183,253,
      253,207,  2,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 39,148,229,253,253,253,
      250,182,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 24,114,221,253,253,253,253,201,
       78,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0, 23, 66,213,253,253,253,253,198, 81,  2,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0, 18,171,219,253,253,253,253,195, 80,  9,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0, 55,172,226,253,253,253,253,244,133, 11,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,136,253,253,253,212,135,132, 16,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
     [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]"""
    array = np.array(eval(array_string))
    # Conv 6 5 1 2
    # ReLU
    # Pool 2 2
    # Conv 12 5 1 0
    # ReLU
    # Pool 2 2
    # Conv 100 5 1 0
    # ReLU
    # Flatten
    # FC 10
    # Softmax
    np.random.seed(42)
    conv_layer1 = ConvolutionalLayer(5, 6, 1, 2)
    relu1 = ReLU()
    pool1 = PollingLayer(2, 2)
    conv2 = ConvolutionalLayer(5, 12, 1, 0)
    relu2 = ReLU()
    pool2 = PollingLayer(2, 2)
    conv3 = ConvolutionalLayer(5, 100, 1, 0)
    flatten = Flatten()
    fc = Dense(10)
    softmax = SoftMax()

    array = array.reshape(1, 28, 28, 1)
    output = conv_layer1.forward(array)
    print(f"{output.shape=}")
    output1 = relu1.forward(output)
    print(f"{output1.shape=}")
    output2 = pool1.forward(output1)
    print(f"{output2.shape=}")
    output3 = conv2.forward(output2)
    print(f"{output3.shape=}")
    output4 = relu2.forward(output3)
    print(f"{output4.shape=}")
    output5 = pool2.forward(output4)
    print(f"{output5.shape=}")
    output6 = conv3.forward(output5)
    print(f"{output6.shape=}")
    output7 = flatten.forward(output6)
    # print(f"{output7=}")
    print(f"{output7.shape=}")
    output8 = fc.forward(output7)
    print(f"{output8.shape=}")
    output9 = softmax.forward(output8)
    print(f"{output9.shape=}")
    print(f"{np.sort(output9, axis=0)=}")


if __name__ == '__main__':
    # main()
    test_forward()
    # main_test_initializer()
    # np.random.seed(42)
    # input_data = np.random.randn(28, 28)  # Example 28x28 input image
    # input_data = input_data.reshape(1, 28, 28, 1)
    # conv_layer = ConvolutionalLayer(num_filters=32, kernel_size=5, stride=1, padding=2)
    # output_data = conv_layer.forward(input_data)[0]
    # print("Output shape:", output_data.shape)

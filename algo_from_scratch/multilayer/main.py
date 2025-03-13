import numpy as np

from AbstractClasses import Layer, Dataset, Activation
from LayerImplementation import ConvolutionalLayer, SoftMax, PollingLayer, Dense, Flatten
from ActivationImplementation import ReLU
from OptiomizerImplementation import SGD
from LossImplementation import CrossEntropyLoss
import matplotlib.pyplot as plt
from algo_from_scratch.multilayer.data_reader import DatasetImages

optimizers = {
    "SGD": SGD,
}
losses = {
    'cross_entropy': CrossEntropyLoss
}


def labels_to_one_hot(labels):
    num_classes = 26  # Total number of classes (uppercase letters)
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        index = ord(label) - ord('A')  # Uppercase letters from 'A' to 'Z'
        one_hot_labels[i, index] = 1
    return one_hot_labels


class Sequential:
    def __init__(self):
        self.layers = list()
        self.data_reader = None
        self.optimizer = None
        self.loss = None

    def add(self, item: (Layer, Activation)) -> None:
        if isinstance(item, (Layer, Activation)):
            self.layers.append(item)
        else:
            raise TypeError(f"Only instances of classes that extend Layer or Activation can be added, "
                            f"but `{type(item).__name__}` was provided.")

    def get(self, index: int) -> Layer:
        """
        Negative indexes like -1 will return last layer, take this into consideration
        :param index: of Layer in the Sequential model
        :return: Layer
        """
        if index < len(self.layers):
            return self.layers[index]
        else:
            raise IndexError("Index out of range")

    def compile(self, optimizer: str = 'SGD', loss='cross_entropy', metrics='accuracy', lr=0.001):
        self.optimizer = optimizers.get(optimizer)(lr=lr)
        self.loss = losses.get(loss)()

    def fit(self, data_reader: Dataset, epochs=10, batch_size=8):
        train_generator = data_reader.get_next_data_row(batch_size=batch_size)
        # ???
        for epoch in range(epochs):
            # y, x
            # looks like list [...], looks like (batch_size, height,width, num channels)
            y_train, x_train = next(train_generator)

            y_predicted = self.forward(x_train)

            # y_hat should be list of list, where each element is one-hot-encoded vector
            # y looks like this already

            y_hat_one_hot_encoded = labels_to_one_hot(y_train)
            loss = self.loss.calculate_loss(y_hat_one_hot_encoded, y_predicted)  # for humans, tracking performance
            print(f"Number of epoch: {epoch}, loss: {loss}")
            loss_deriv = self.loss.backward(y_hat_one_hot_encoded, y_predicted)  # for model param updating

            self.backward(loss_deriv)

            # for batch in range(batch_size):
            #     print(f'(Training) Epoch: {epoch + 1} -> {batch + 1}/{batch_size} Batches Trained.', end='\r')

    def evaluate(self):
        pass

    def backward(self, do):
        deriv_output = do
        # deriv_list = []
        for layer in reversed(self.layers):
            if isinstance(layer, SoftMax):
                continue

            deriv_output = layer.backward(deriv_output)


    def forward(self, Input):
        output = Input
        out_list = []
        # counter = 1
        for layer in self.layers:
            output = layer.forward(output)
            # out_list.append(output)
            # print(f"{counter} {output.shape = }")
            # counter += 1
        # return output, out_list
        return output

    def get_info(self):
        info = ''
        for layer in self.layers:
            info += layer.__str__()
            info += '\n'
        return info


def plot_layers(output):
    def plot_feature_maps(feature_maps, title):
        num_filters = feature_maps.shape[2]
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters * 2, 2))
        fig.suptitle(title)

        if num_filters == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.imshow(feature_maps[:, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()

    # Plot the outputs
    plot_feature_maps(output[0][0], "Output of Conv Layer 1")
    plot_feature_maps(output[1][0], "Output of ReLU 1")
    plot_feature_maps(output[2][0], "Output of Pool 1")
    plot_feature_maps(output[3][0], "Output of Conv Layer 2")
    plot_feature_maps(output[4][0], "Output of ReLU 2")
    plot_feature_maps(output[5][0], "Output of Pool 2")
    plot_feature_maps(output[6][0], "Output of Conv Layer 3")

    # Since output7, output8, and output9 are flattened or 1D, we plot them directly
    plt.figure(figsize=(12, 4))
    plt.plot(output[7])
    plt.title("Output of Flatten Layer")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(output[8])
    plt.title("Output of Fully Connected Layer")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(output[9])
    plt.title("Output of SoftMax Layer")
    plt.show()

def normal_test():
    model_param = {
        'l1': (ConvolutionalLayer, (5, 6, 1, 2)),
        'l2': (ReLU, ()),
        'l3': (PollingLayer, (2, 2)),
        "l4": (ConvolutionalLayer, (5, 12, 1, 0)),
        'l5': (ReLU, ()),
        'l6': (PollingLayer, (2, 2)),
        'l7': (ConvolutionalLayer, (5, 100, 1, 0)),
        'l8': (Flatten, ()),
        'l9': (Dense, (26,)),
        'l10': (SoftMax, ())
    }
    np.random.seed(42)

    model = Sequential()
    for key, value in model_param.items():
        model.add(value[0](*value[1]))

    model.compile()
    data_reader = DatasetImages("dataset", 0.2)
    model.fit(data_reader)

def test():
    model_param = {
        'l1': (ConvolutionalLayer, (5, 6, 1, 2)),
        'l2': (ReLU, ()),
        'l3': (PollingLayer, (2, 2)),
        "l4": (ConvolutionalLayer, (5, 12, 1, 0)),
        'l5': (ReLU, ()),
        'l6': (PollingLayer, (2, 2)),
        'l7': (ConvolutionalLayer, (5, 100, 1, 0)),
        'l8': (Flatten, ()),
        'l9': (Dense, (10, )),
        'l10': (SoftMax, ())
    }
    np.random.seed(42)

    model = Sequential()

    for key, value in model_param.items():
        model.add(value[0](*value[1]))

    my_generator = DatasetImages("dataset", 0.2)
    getter = my_generator.get_next_data_row(batch_size=2)
    batch = [next(getter)]

    batch_number = 0
    letter = 0
    data_img = 1

    image_data = batch[batch_number][data_img][0]
    image_data_shape = image_data.shape
    image_data_1 = image_data.reshape(1,image_data_shape[1], image_data_shape[0], 1)

    # image_data2 = batch[batch_number][data_img][1]
    # image_data_shape2 = image_data2.shape
    # image_data_2 = image_data2.reshape(1,image_data_shape2[1], image_data_shape2[0], 1)
    # batched_array = np.concatenate((image_data_1, image_data_2), axis=0)

    output = model.forward(image_data_1)
    first_batch_out = output[0]
    second_batch_out = output[0][1]

    y_true_1 = np.array([0,0,1,0,0,0,0,0,0,0])

    print(y_true_1)
    print(first_batch_out)
    loss1 = CrossEntropyLoss().calculate_loss(y_true_1, [first_batch_out])
    loss_der = CrossEntropyLoss().backward(y_true_1, [first_batch_out])

    model.backward(loss_der)

    # y_true_2 = [0]*10
    # y_true_2[2] = 1
    #
    # loss2 = CrossEntropyLoss().calculate_loss([y_true_2], [second_batch_out])
    # print(f"loss of first batch: {loss1}")
    # print(f"loss of second batch: {loss2}")
    # print(f"average loss in  batch: {(loss1+loss2)/2}")
    #
    # batched_y_true = np.array([y_true_1, y_true_2])
    # batched_out = output[0]
    # print(f"shape of batched y_true = {batched_y_true.shape}")
    # print(f"shape of batched predict = {batched_out.shape}")
    # print(f"Loss of the batch: {CrossEntropyLoss().calculate_loss(batched_y_true, batched_out)}")

    # print(np.sum(output[0]))
    # print(model.get_info())

    # plot_layers([output, output1, output2, output3, output4, output5, output6, output7, output8, output9])
    # plot_layers(output[1])


if __name__ == '__main__':
    normal_test()

import argparse
import csv
import math
import random
from typing import Union

import matplotlib.pyplot as plt


# utils
def minus_of_list(x1: list, x2: list) -> list:
    return [x - y for x, y in zip(x1, x2)]


def sum_of_list(x1: list, x2: list) -> list:
    return [x + y for x, y in zip(x1, x2)]


def power_of_list(x: list, power) -> list:
    return [el ** power for el in x]


def product_of_list(x1: list, x2: list) -> list:
    return [x * w for x, w in zip(x1, x2)]


def dot_product(x1: list[float], w1: list[float]):
    return sum(product_of_list(x1, w1))


def divide_list_by_one_element(x1, element):
    return [x/element for x in x1]


# activation functions
def linear(x, is_derivative=False):
    if is_derivative:
        return 1
    return x


class Layer:
    def __init__(self, n_neurons: int, n_weights: int, activation_function: callable):
        self.n_neurons: int = n_neurons
        self.activation_function: callable = activation_function
        # self.weights: list[list[float]] = create_weights(n_neurons, n_weights)
        self.weights: list[list[float]] = [n_weights*[0]]
        self.bias = [0] * n_neurons

    def calculate_output(self, vector_in: list[Union[float]]) -> list[float]:
        net_values: list[float] = []
        for index in range(self.n_neurons):
            current_weights = self.weights[index]
            current_bias = self.bias[index]
            net_value = dot_product(current_weights,vector_in) - current_bias
            net_values.append(net_value)

        return net_values

    def forward_propagation(self, vector_in):
        out_current_layer: list[float] = self.calculate_output(vector_in)
        activated_output: list[float] = [self.activation_function(net_value) for net_value in out_current_layer]
        return activated_output

    def backward_propagation(self):
        # no implementation currently
        pass

    def update_weights(self, new_weights):
        self.weights = new_weights

    def update_bias(self, new_bias):
        self.bias = new_bias


class NeuralNetwork:
    def __init__(self, layer, n_epochs: int, value_map, learning_rate=0.1):
        # layers should be a list of layer and activation functions after or i can assign activation func to each of
        # them directly i will go first with assigning it directly first
        # self.layers: list[Layer] = list(args)
        self.layer: Layer = layer
        self.initialize_weights()
        self.x_train = None
        self.x_train = None
        self.y_train = None
        self.learning_rate = learning_rate
        self.number_of_epochs = n_epochs
        self.value_map = value_map

    def train(self):
        accuracy = []

        for epoch in range(self.number_of_epochs):
            # features 26 dim vector, target is the index of correct answer in a list [0,0,0]
            for (features, target) in zip(self.x_train, self.y_train):
                number_target = self.value_map.get(target)
                predicted_output = self.layer.forward_propagation(features)
                errors = [0]*self.layer.n_neurons # [0,0,0]
                errors[number_target] = 1
                errors = minus_of_list(errors,predicted_output)
                # now we will have something like [-x, +x, -x] so that only correct neuron gets better weights,
                # while other get lesser

                # adjustment of the weights
                new_weights_for_layer = []
                new_biases_for_layer = []
                for index in range(len(errors)):
                    coefficient = errors[index] * self.learning_rate
                    calculated_vector = [coefficient * feature for feature in features]
                    calculated_vector.append(coefficient*-1)

                    bias = self.layer.bias[index]
                    old_weights = self.layer.weights[index]
                    old_weights.append(bias)
                    new_weights = sum_of_list(old_weights, calculated_vector)
                    new_weights_for_layer.append(new_weights[:-1])
                    new_biases_for_layer.append(new_weights[-1])

                self.layer.update_bias(new_biases_for_layer)
                self.layer.update_weights(new_weights_for_layer)

            # calculation of loss for plot later for back_propagation implementation
            current_acc = self.test_accuracy()
            accuracy.append(current_acc)

        return accuracy

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        return self.train()

    def predict(self, input) -> int:
        # returns an index of classified language
        # input 26dim vector
        predictions = self.layer.forward_propagation(input)
        return predictions.index(max(predictions))

    def predict_across(self):
        return [self.predict(line) for line in self.x_test]

    def initialize_weights(self):
        input_dim = len(self.layer.weights[0])
        output_dim = self.layer.n_neurons
        self.layer.update_weights(create_weights(output_dim, input_dim))
        # for layer in self.layers:
        #     input_dim = len(layer.weights)
        #     output_dim = layer.n_neurons
        #     layer.update_weights(create_weights(input_dim, output_dim))

    def test_accuracy(self):
        current_preds = self.predict_across()
        correct_answers = 0
        for index, line in enumerate(self.y_test):
            if self.value_map.get(line) == current_preds[index]:
                correct_answers+=1
        return correct_answers/len(self.y_test)


def create_weights(n_neurons: int, n_weights: int) -> list[list[float]]:
    # i use Xavier/Glorot initialization of weights
    list_of_weights: list[list[float]] = []
    for neuron in range(n_neurons):
        weights_for_neuron = []

        for _ in range(n_weights):
            weights_for_neuron.append(
                random.uniform(
                    -math.sqrt(6/n_neurons+n_weights),
                    math.sqrt(6 / n_neurons + n_weights)
                )
            )

        list_of_weights.append(weights_for_neuron)

    return list_of_weights


def read_data_from_file(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        return [line for line in reader]


def convert_target_to_number(targets):
    return {
        target: index
        for index, target in enumerate(targets)
    }


def train_test_split(file_path, test_size):
    raw_data = read_data_from_file(file_path)
    test_size = int(len(raw_data) * test_size)

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for line in raw_data[:-test_size]:
        x_train.append(vectorize(line[1]))
        y_train.append(line[0])

    for line in raw_data[-test_size:]:
        x_test.append(vectorize(line[1]))
        y_test.append(line[0])

    return x_train, x_test, y_train, y_test


def vectorize(text_string):
    text_string = text_string.lower()
    output_vector = 26*[0]
    for char in text_string:
        if 'a' <= char <= 'z':
            output_vector[ord(char)-ord('a')]+=1

    return divide_list_by_one_element(output_vector, sum(output_vector))


def default(args):
    x_train, x_test, y_train, y_test = train_test_split(args.dataset_path, test_size=0.2)
    n_epochs = args.e
    learning_rate = args.alpha
    value_map = convert_target_to_number(set(y_train))
    nnetwork = NeuralNetwork(Layer(len(value_map), 26, linear), n_epochs, value_map, learning_rate)
    accuracies = nnetwork.fit(x_train,y_train, x_test, y_test)
    print(f"Accuracy of the model is: {nnetwork.test_accuracy()*100:.2f}%")
    plt.plot(range(1, n_epochs + 1), accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per epoch")
    plt.show()


def iterate(args):
    x_train, x_test, y_train, y_test = train_test_split(args.dataset_path, test_size=0.2)
    n_epochs = args.e
    learning_rate = args.alpha
    value_map = convert_target_to_number(set(y_train))
    nnetwork = NeuralNetwork(Layer(len(value_map), 26, linear), n_epochs, value_map, learning_rate)
    nnetwork.fit(x_train,y_train, x_test, y_test)
    languages_to_classify = list(value_map.keys())
    print('\n')
    print("I can only classify those classes:", *languages_to_classify)
    print('\n')
    user_stopped = False

    while not user_stopped:
        user_input = input(f"Please give me text in either {languages_to_classify} languages:\nIn any other situation skip current "
                           f"iteration.\nType"
                           f"'stop' to end classification\n> ")

        if user_input == 'stop':
            print('End of classification')
            break

        try:
            vector = vectorize(user_input)
            predicted_as = nnetwork.predict(vector)
            predicted_label = [key for key, val in value_map.items() if val == predicted_as]
            print(f'\nPrediction of perceptron is: {predicted_as}')
            print(f'Which means: {predicted_label[0]}\n')

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    #  For reference i use uniform weights, hence 6/(fanin+fanout)
    parser = argparse.ArgumentParser(description="NN with 1 layer from scratch")
    parser.add_argument("-ds", "--dataset-path", type=str,
                        help="Provide a path to a dataset from current folder. Otherwise, give full path",
                        required=True)

    parser.add_argument("-e", type=int, help="Number of epochs",
                        required=False, default=30)
    parser.add_argument("-a", "--alpha", type=float, help="Constant learning rate",
                        required=False, default=0.1)
    parser.add_argument("-it",
                        "--iterate-mode",
                        type=int,
                        help="Iterative mode, you give me vectors I give you classes",
                        choices=[0, 1],
                        required=False,
                        default=0)

    args = parser.parse_args()

    if args.dataset_path is  None:
        raise Exception('You should provide path to datasets.')

    if args.iterate_mode == 1:
        iterate(args)
        exit(0)

    default(args)

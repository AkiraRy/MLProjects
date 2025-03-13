import math
import argparse
import random
import sys


# File read. test/train split
def read_file(url):
    with open(url, 'r') as f:
        file = f.readlines()
    return [line.strip("\n").split(",") for line in file]


def train_test_split(train_path, test_path=None):
    training_lines = read_file(train_path)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for line in training_lines:
        x_train.append([float(el) for el in line[:4]])
        y_train.append(line[-1])

    if not test_path:
        return x_train, y_train

    test_lines = read_file(test_path)

    for line in test_lines:
        x_test.append([float(el) for el in line[:4]])
        y_test.append(line[-1])
    return x_train, x_test, y_train, y_test


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


def get_random_weights(number_of_weights):
    return [random.uniform(-5, 5) for _ in range(number_of_weights)]


class Perceptron:
    slots = (
        'weights',
        'theta',
        'learning_rate',
        'number_of_epochs',
        'x_train',
        'y_train',
        'value_map'
    )

    def __init__(self, map_value, learning_rate=0.1, number_of_epochs=10, ):
        self.x_train = None
        self.y_train = None
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.weights = []
        self.theta = 1  # by default
        self.value_map = map_value

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.weights = get_random_weights(len(x_train[0]))
        self.theta = random.uniform(0, 1)
        self.train()

    def update_weights(self, new_weights):
        assert self.weights != new_weights, 'same weight'
        self.weights = new_weights

    def update_theta(self, new_theta):
        assert self.theta != new_theta, 'same theta'
        self.theta = new_theta

    def activation(self, vector: list[float]):
        net_value = dot_product(vector, self.weights)
        return 1 if net_value >= self.theta else 0

    def train(self):
        for epoch in range(self.number_of_epochs):
            for (features, target) in zip(self.x_train, self.y_train):
                predicted_output = self.activation(features)
                real_output = self.value_map.get(target)
                self.update_neuron(predicted_output, real_output, features)

    def update_neuron(self, actual_decision, correct_decision, x_t):
        if correct_decision == actual_decision:
            return

        # Right side of the equation
        coefficient = (correct_decision - actual_decision) * self.learning_rate

        calculated_vector = [coefficient * number for number in x_t]
        calculated_vector.append(coefficient * -1)
        # Left side
        old_weights = self.weights
        old_weights.append(self.theta)
        # Finale
        new_weights = sum_of_list(old_weights, calculated_vector)
        # Updating
        self.update_weights(new_weights[:-1])
        self.update_theta(new_weights[-1])

    def predict(self, x_test):
        return [self.activation(row) for row in x_test]

    def solo_predict(self, vector):
        return self.predict([vector])


def accuracy_score_for_perceptron(y_preds, y_test, mapped_value):
    mapped_y_test = map(lambda x: mapped_value[x], y_test)
    accuracy_list = [1 if y_t == y_y else 0 for y_t, y_y in zip(mapped_y_test, y_preds)]
    accuracy_percent = accuracy_list.count(1) / len(accuracy_list)
    return accuracy_percent * 100


def accuracy_score(y_test, y_preds):
    assert len(y_test) == len(y_preds), 'Different shapes, error'
    accuracy_list = [1 if y_t == y_y else 0 for y_t, y_y in zip(y_test, y_preds)]
    accuracy_percent = accuracy_list.count(1) / len(accuracy_list)
    return accuracy_percent * 100


def convert_target_to_binary(targets):
    targets = list(targets)
    assert len(targets) == 2, 'Cant classify unless there is only 2 classes'
    return {
        targets[0]: 0,
        targets[1]: 1
    }


def accuracy_score_for_bin_class(y_preds, y_test, map_value):
    keys = list(map_value.keys())
    first_class_name = keys[0]
    first_class_indexes_test = [i for i, value in enumerate(y_test) if value == first_class_name]

    first_class_preds = [y_preds[i] for i in first_class_indexes_test]
    first_class_correct_values = [map_value.get(y_test[i]) for i in first_class_indexes_test]
    first_class_score = accuracy_score(first_class_correct_values, first_class_preds)

    second_class_name = keys[1]
    second_class_indexes_test = [i for i, value in enumerate(y_test) if value == second_class_name]

    second_class_preds = [y_preds[i] for i in second_class_indexes_test]
    second_class_correct_values = [map_value.get(y_test[i]) for i in second_class_indexes_test]
    second_class_score = accuracy_score(second_class_correct_values, second_class_preds)

    return first_class_score, first_class_name, second_class_score, second_class_name


def default(args):
    x_t, x_test, y_t, y_test = train_test_split(args.train_path, args.test_path)
    number_of_epochs = args.e or 5
    learning_rate = args.alpha or 0.1
    map_value = convert_target_to_binary(set(y_t))
    perceptron = Perceptron(number_of_epochs=number_of_epochs, learning_rate=learning_rate, map_value=map_value)
    perceptron.fit(x_t, y_t)
    y_preds = perceptron.predict(x_test)

    first_class_score, first_class_name, second_class_score, second_class_name = accuracy_score_for_bin_class(y_preds, y_test, map_value)

    print(f"accuracy score overall: {accuracy_score_for_perceptron(y_preds, y_test, map_value)}%")
    print(f"\nAccuracy per class is:\n{first_class_name}: {first_class_score}%\n{second_class_name}: {second_class_score}%")


def iterate(args):
    x_t, y_t = train_test_split(args.train_path)
    number_of_epochs = args.e or 5
    learning_rate = args.alpha or 0.1
    map_value = convert_target_to_binary(set(y_t))
    perceptron = Perceptron(number_of_epochs=number_of_epochs, learning_rate=learning_rate, map_value=map_value)
    perceptron.fit(x_t, y_t)
    print('\n')
    print("I can only classify those classes:", *list(map_value.keys()))
    print('\n')
    user_stopped = False

    while not user_stopped:
        user_input = input(f"Please give me 4 values separated by ','.\nIn any other situation skip current "
                           f"iteration.\nType"
                           f"'stop' to end classification\n> ")

        if user_input == 'stop':
            print('End of classification')
            break

        try:
            vector = [float(value) for value in user_input.split(',')]
            predicted_as = perceptron.solo_predict(vector)
            print(f'Prediction of perceptron is: {predicted_as}')
            print(f'Which means: {map_value.get(predicted_as)}')

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KNN from scratch")
    parser.add_argument("-trp", "--train-path", type=str,
                        help="Provide a path to a train dataset from current folder. Otherwise, give full path",
                        required=False)
    parser.add_argument("-tp", "--test-path", type=str,
                        help="Provide a path to a test dataset from current folder. Otherwise, give full path",
                        required=False)
    parser.add_argument("-e", type=int, help="Number of epochs", required=False, default=None)
    parser.add_argument("-a", "--alpha", type=float, help="Constant learning rate", required=False, default=None)
    parser.add_argument("-it",
                        "--iterate-mode",
                        type=int,
                        help="Iterative mode, you give me vectors I give you classes",
                        choices=[0, 1],
                        required=False,
                        default=0)

    args = parser.parse_args()

    if args.train_path is None or args.test_path is None:
        raise Exception('You should provide path to datasets.')

    if args.iterate_mode == 1:
        iterate(args)
        exit(0)

    default(args)

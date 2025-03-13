import math
import argparse
import sys
import matplotlib.pyplot as plt


def minus_of_list(x1: list, x2: list) -> list:
    return [x - y for x, y in zip(x1, x2)]


def power_of_list(x: list, power):
    return [el**power for el in x ]


def euclidean_distance(x1, x2):
    return math.sqrt(sum(power_of_list(minus_of_list(x1, x2), 2)))


class KNN:
    def __init__(self, k: int = 3):
        self.y_train = None
        self.x_train = None
        self.k = k

    def fit(self, x_train, y_train):
        if any(len(vec) != len(x_train[0]) for vec in x_train):
            raise ValueError("Inconsistent dimensions in the training data")
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        y_preds = []
        for row in x_test:
            nearest_neighbours = self.prediction(row)
            y_preds.append(nearest_neighbours)
        return y_preds

    def prediction(self, X):
        distances = list()

        for (x_train, y_train) in zip(self.x_train, self.y_train):
            distance = euclidean_distance(x_train, X)
            distances.append((distance, y_train))

        distances.sort(key=lambda x: x[0])
        closest = list()
        for i in range(self.k):
            closest.append(distances[i][1])

        most_common = max(set(closest), key=closest.count)

        return most_common


def read_file(url):
    with open(url, 'r') as f:
        file = f.readlines()
    list_data = [line.strip("\n").split(",") for line in file]
    return list_data


def train_test_split(train_path, test_path):
    training_lines = read_file(train_path)
    test_lines = read_file(test_path)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for line in training_lines:
        x_train.append([float(el) for el in line[:4]])
        y_train.append(line[-1])

    for line in test_lines:
        x_test.append([float(el) for el in line[:4]])
        y_test.append(line[-1])

    return x_train, x_test, y_train, y_test


def accuracy_score(y_test, y_preds):
    assert len(y_test) == len(y_preds), 'Different shapes, error'

    accuracy_list = [1 if y_t == y_y else 0 for y_t, y_y in zip(y_test, y_preds)]
    accuracy_percent = accuracy_list.count(1) / len(accuracy_list)
    return accuracy_percent * 100


def plot_all_of_them(args):
    x_train, x_test, y_train, y_test = train_test_split(args.train_path, args.test_path)

    plot_data = []
    for i in range(1, 106):
        knn = KNN(i)
        knn.fit(x_train, y_train)
        y_preds = knn.predict(x_test)
        plot_data.append(accuracy_score(y_test, y_preds))

    plt.plot(plot_data)
    plt.ylabel("Accuracy")
    plt.xlabel("Value of \"k\"")
    plt.show()


def main(args) -> None:
    x_train, x_test, y_train, y_test = train_test_split(args.train_path, args.test_path)
    knn = KNN(args.k)
    knn.fit(x_train, y_train)
    y_preds = knn.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, y_preds):.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KNN from scratch")
    parser.add_argument("-trp","--train-path", type=str, help="Provide a path to a train dataset from current folder. Otherwise, give full path", required=False)
    parser.add_argument("-tp","--test-path", type=str, help="Provide a path to a test dataset from current folder. Otherwise, give full path", required=False)
    parser.add_argument("-k", type=int, help="Number of nearest neighbours", required=False, default=3)
    parser.add_argument("-p", '--plot', type=int, help="Plot from k=1 to k=105. provide p>0 to use this option", required=False, default=0)

    args = parser.parse_args()

    if args.train_path  is None or args.test_path is None:
        raise Exception('You should provide path to datasets.')

    if args.plot > 0:
        plot_all_of_them(args)
        sys.exit()

    main(args)
    # traning = 'train.csv'
    # test = 'test.csv'
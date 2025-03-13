import csv
import random

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.01):
        # The smaller the learning rate, the slower the convergence, but the more accurate the results!
        self.learning_rate = learning_rate
        # Weights set to None since they will be initialized during the learning process
        self.weights = None

    @staticmethod
    def load_data(file_path):
        # To store data from the file
        data = []
        # Map language names to numeric values — easier to work with
        language_mapping = {"English": 0, "German": 1, "Polish": 2, "Spanish": 3}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Read from the file
            reader = csv.reader(csvfile)
            # Iterate over rows
            for row in reader:
                # Split into language and text
                language, text = row[0], row[1]
                # Remove unwanted quotes
                text = text.strip('"')
                language = language_mapping[language.strip().strip('"')]
                # Add to data as a tuple
                data.append((text, language))
        return data

    @staticmethod
    def preprocess(text):
        # Convert text to lowercase
        text = text.lower()
        # Array to store the frequency of each letter
        letter_count = np.zeros(26)
        for letter in text:
            # Work with ASCII codes
            if 'a' <= letter <= 'z':
                # Increment letter count
                letter_count[ord(letter) - ord('a')] += 1

        # We want to get a vector that always has a length of 1.
        # This maintains the frequency of characters while normalizing the vector length.
        # This ensures the text length reflects in the results.
        # We use L2 norm — sqrt(sum of squares)
        normalization = letter_count / np.linalg.norm(letter_count)
        return normalization

    def activation(self, values):
        # dot — inner product a1b1 + a2b2 ...
        return np.dot(self.weights, values)

    def classify(self, row):
        # Prepare an array with letter occurrences
        # Adding bias at the beginning
        inputs = np.array([1] + list(self.preprocess(row)))

        # Get the activation level
        activations = self.activation(inputs)

        # To classify the language, we select the perceptron with the highest activation.
        return np.argmax(activations)

    def train(self, training_data, test_data, num_epochs=100, num_languages=4):
        # Initialize weights — one for each language (in this case, 4) with 26 letters

        # Set a random weight between 0 and 1 for the first weight
        # This maintains relative neutrality while avoiding local minima
        self.weights = np.array([[random.uniform(0, 1)] + [0.0] * 26 for _ in range(num_languages)])

        accuracies = []

        for epoch in range(num_epochs):
            for row, expected_output in training_data:
                # Preprocessing — add 1 at the beginning as bias
                inputs = np.array([1] + list(self.preprocess(row)))

                # Calculate the activation function
                obtained_output = self.activation(inputs)
                # Set errors to 0
                error = np.zeros(num_languages)
                # Set 1 for the expected language
                error[expected_output] = 1
                # Calculate the difference between expected and obtained inputs
                error -= obtained_output
                # Update perceptron weights
                self.weights += self.learning_rate * np.outer(error, inputs)

            # Count correct classifications
            correct_classifications = sum(1 for row, language in test_data if self.classify(row) == language)
            accuracy = correct_classifications / len(test_data)
            accuracies.append(accuracy)

        return accuracies


def main():
    learning_rate = float(input("Enter the learning rate (for best results, use a small value around 0.01): "))
    num_epochs = int(input("Enter the number of epochs: "))

    perceptron = Perceptron(learning_rate=learning_rate)
    training_data = perceptron.load_data("lang.train.csv")
    test_data = perceptron.load_data("lang.test.csv")

    accuracies = perceptron.train(training_data, test_data, num_epochs=num_epochs)

    # Plot
    plt.plot(range(1, num_epochs + 1), accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.show()

    correct_classifications = sum(1 for row, language in test_data if perceptron.classify(row) == language)
    accuracy = correct_classifications / len(test_data)
    print(f'Classification accuracy: {accuracy}')

    mapping = {0: "English", 1: "German", 2: "Polish", 3: "Spanish"}

    while True:
        user_input = input("Enter text to classify, or type 'q' to quit: ")
        if user_input.lower() == "q":
            break
        result = perceptron.classify(user_input)
        print(f'Classified as language: {mapping[result]}')


if __name__ == '__main__':
    main()

# returns a dictionary containing paths(value) to each letter (key)
import math
import os
from typing import Union, Tuple, List
from matplotlib import pyplot as plt
from AbstractClasses import Dataset
from PIL import Image
import numpy as np


def get_image_paths(folder_path):
    image_paths_dict = {}
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) and item_path.lower().endswith('.png'):
            parent_folder = os.path.basename(folder_path)
            if parent_folder not in image_paths_dict:
                image_paths_dict[parent_folder] = []
            image_paths_dict[parent_folder].append(item_path)
        elif os.path.isdir(item_path):
            nested_paths = get_image_paths(item_path)
            for key, value in nested_paths.items():
                image_paths_dict.setdefault(key, []).extend(value)
    return image_paths_dict


def convert_paths_to_array(image_paths: list):
    converted_paths = []
    for img_path in image_paths:
        image_data = Image.open(img_path).convert("L")
        image_data1 = image_data.resize((28, 28), Image.LANCZOS)
        image_array = np.asarray(image_data1) / 255
        image_array = image_array.reshape((28, 28, 1))
        converted_paths.append(image_array)
    np.random.shuffle(converted_paths)
    return converted_paths


def get_numpy_arrays(dict_paths: dict[str, list[str]]):
    dict_numpy = dict()
    for key, value in dict_paths.items():
        new_value = convert_paths_to_array(value)
        dict_numpy.setdefault(key, []).extend(new_value)

    return dict_numpy

# also another problem should i shuffle all those value lists for each key or shuffle paths instead of numpy arrays
#  and should i shuffle list of keys?
# i mean so that we always go in different orders? but the length is still the same
# so do i shuffle list of keys every 7 iterations?
# would be better to have somewhere some strategy for sorting it

# i cant just shuffle, because min index or max index depends on the posotion of keys
# or i can just make a tuple of them like (key, min/max_index) and shuffle tuples in a list instead
# so that i can add some randomness


class DatasetImages(Dataset):
    __slots__ = (
        "numpy_dict",
        "test_set_percentage",
        "batch_size",
        "max_index_for_each_key_train",
        "list_of_keys",
        "max_len",
    )

    def __init__(self, dataset_path, test_set_percentage):
        super().__init__(dataset_path, test_set_percentage)
        self.numpy_dict = get_numpy_arrays(get_image_paths(self.dataset_path))
        self.list_of_keys = list(self.numpy_dict.keys())
        self.max_len = 0
        self.initialize_max_values()

    def get_next_test_row(self, batch_size=1) -> Tuple[List[str], np.ndarray]:
        current_index = 0  # corresponds to the letter
        current_value_index = min(self.max_index_for_each_key_train)  # corresponds to the image value in an array

        while current_value_index < self.max_len:
            batch_keys = []
            batch_values = []

            for _ in range(batch_size):
                if current_index % len(self.list_of_keys) == 0 and current_index != 0:
                    current_index = 0
                    current_value_index += 1

                key = self.list_of_keys[current_index]
                values_for_key = self.numpy_dict.get(key)

                if current_value_index < self.max_index_for_each_key_train[current_index]:
                    current_index += 1
                    continue

                if not current_value_index < len(values_for_key):
                    current_index += 1
                    continue

                batch_keys.append(key)
                batch_values.append(values_for_key[current_value_index])
                current_index += 1

            if batch_keys:
                yield batch_keys, np.array(batch_values)

    def get_next_data_row(self, batch_size=1) -> Tuple[List[str], np.ndarray]:
        current_index = 0  # corresponds to the letter
        current_value_index = 0  # corresponds to the image value
        max_value_index = max(self.max_index_for_each_key_train)

        while current_value_index < max_value_index:
            batch_keys = []
            batch_values = []

            for _ in range(batch_size):
                if current_index % len(self.list_of_keys) == 0 and current_index != 0:
                    current_index = 0
                    current_value_index += 1

                key = self.list_of_keys[current_index]
                values_for_key = self.numpy_dict.get(key)

                if current_value_index >= self.max_index_for_each_key_train[current_index]:
                    current_index += 1
                    continue

                if not current_value_index < len(values_for_key):
                    current_index += 1
                    continue
                batch_keys.append(key)
                batch_values.append(values_for_key[current_value_index])
                current_index += 1
            if batch_keys:
                yield batch_keys, np.array(batch_values)

    def initialize_max_values(self):
        self.max_index_for_each_key_train = []

        for _, value in self.numpy_dict.items():
            actual_value_len = len(value)
            if actual_value_len > self.max_len:
                self.max_len = actual_value_len

            allowed_len = 1 - self.test_set_percentage
            self.max_index_for_each_key_train.append(math.ceil(actual_value_len * allowed_len))

# from collections import Counter


if __name__ == '__main__':
    # Testing if this thing works correctly

    my_generator = DatasetImages("dataset", 0.2)

    generator = my_generator.get_next_data_row(batch_size=4)
    cos = next(generator)
    print(cos[0])
    print(cos[1].reshape(4,28,28,1).shape)
    # getter = my_generator.get_next_data_row()
    # batch = [next(getter)]
    #
    # batch_number = 0
    # letter = 0
    # data_img = 1
    #
    # image_data = batch[batch_number][data_img][0]
    # print(image_data.shape)
    # image_data_shape = image_data.shape
    # image_data_1 = image_data.reshape(1, image_data_shape[1], image_data_shape[0], 1)
    #
    # plt.imshow(image_data_1[0], cmap='gray')  # cmap='gray' for black and white images
    # plt.axis('off')  # Turn off axis
    # plt.show()

# def numpy_to_image(array):
    #     array = (array * 255).astype(np.uint8)
    #     if array.shape[2] == 1:
    #         array = array.squeeze(axis=2)
    #         return Image.fromarray(array, mode='L')
    #     else:
    #         return Image.fromarray(array, mode='RGB')
    #
    #
    # mine_image_pooled = max_pool_mine.forward(image_data)[0]
    # his_image_pooled = max_pool_his.forward(image_data)[0]
    # print(his_image_pooled.shape)
    # # Convert the original and pooled images to PIL Images
    # original_image = numpy_to_image(image_data[0])
    # mine_pooled_image = numpy_to_image(mine_image_pooled)
    # his_pooled_image = numpy_to_image(his_image_pooled)
    #
    # # Plot the images using matplotlib
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #
    # ax[0].imshow(original_image)
    # ax[0].set_title("Original Image")
    # ax[0].axis('off')
    #
    # ax[1].imshow(mine_pooled_image)
    # ax[1].set_title("Mine Pooled Image")
    # ax[1].axis('off')
    #
    # ax[2].imshow(his_pooled_image)
    # ax[2].set_title("His Pooled Image")
    # ax[2].axis('off')
    #
    # plt.show()

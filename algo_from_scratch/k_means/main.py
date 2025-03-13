from collections import defaultdict

import pandas as pd
import numpy as np


def vector_to_cluster(centroids, vector):
    # returns index of cluster to which the vector is closest
    min_distance = float('inf')
    index_of_centroid = -1

    for index, centroid in enumerate(centroids):
        distance = np.linalg.norm(vector - centroid)

        if distance < min_distance:
            min_distance = distance
            index_of_centroid = index

    return index_of_centroid, min_distance


def k_means(data, k, max_n_iterations=1000):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for i in range(max_n_iterations):
        new_centroids = []
        cluster_indexes = []
        sum_of_distances = 0
        sum_of_distances_per_cluster = defaultdict(float)

        for vector in data:
            index_of_centroid, min_distance = vector_to_cluster(centroids, vector)
            cluster_indexes.append(index_of_centroid)
            sum_of_distances += min_distance
            sum_of_distances_per_cluster[index_of_centroid] += min_distance

        labels = np.array(cluster_indexes)  # aka index of a group

        for key, value in sum_of_distances_per_cluster.items():
            print(f"Iteration {i + 1}: cluster: {key} sum_sq_distances: {value}")

        # new centroids
        for j in range(k):
            cluster = data[labels == j]
            centroid = cluster.mean(axis=0)
            new_centroids.append(centroid)

        new_centroids = np.array(new_centroids)
        print(f'Iteration {i + 1}: total sum: {sum_of_distances:.2f}')

        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids

    return labels, centroids


if __name__ == "__main__":
    data = pd.read_csv('data.csv', header=None)

    # all rows, columns except last one
    input_data = data.iloc[:, :-1].values
    n_k = int(input("give me number of k"))

    labels, centroids = k_means(input_data, n_k)
    print(centroids)
    print(labels)

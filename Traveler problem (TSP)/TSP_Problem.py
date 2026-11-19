import numpy as np


def generate_cities(n_cities=25, seed=42, size=100):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, size, size=(n_cities, 2))


def distance_matrix(cities):
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def tour_length(tour, dist_matrix):
    return np.sum(dist_matrix[tour, np.roll(tour, -1)])
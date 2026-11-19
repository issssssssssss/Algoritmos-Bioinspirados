
import numpy as np

evaluation_count = 0
evaluation_history = []

DEFAULT_ITEMS = [
    (2, 3), (3, 4), (4, 5), (5, 6), (9, 10),
    (7, 8), (1, 2), (6, 7), (8, 9), (3, 5)
]
DEFAULT_CAPACITY = 20


def decode_position(x, threshold=0.5):
    arr = np.asarray(x)
    if arr.dtype.kind in {"i", "u", "b"}:
        return arr.astype(int)
    return (arr >= threshold).astype(int)


def fitness(individual, items=None, capacity=None, decode_fn=None, threshold=0.5):
    if items is None:
        items = DEFAULT_ITEMS
    if capacity is None:
        capacity = DEFAULT_CAPACITY
    if decode_fn is None:
        individual = decode_position(individual, threshold=threshold)
    else:
        individual = np.asarray(decode_fn(individual), dtype=int)
    weight = sum(items[i][0] for i in range(len(items)) if individual[i] == 1)
    value = sum(items[i][1] for i in range(len(items)) if individual[i] == 1)
    if weight <= capacity:
        return -float(value)
    return 1000.0 + float(weight - capacity)


def evaluate(x, items=None, capacity=None, decode_fn=None, threshold=0.5):
    global evaluation_count
    evaluation_count += 1
    return fitness(x, items, capacity, decode_fn=decode_fn, threshold=threshold)


def evaluate_population(population, items=None, capacity=None, decode_fn=None, threshold=0.5):
    return np.asarray([
        evaluate(x, items, capacity, decode_fn=decode_fn, threshold=threshold)
        for x in population
    ])


def result_position(x, threshold=0.5):
    return decode_position(x, threshold=threshold)


def reset():
    global evaluation_count, evaluation_history
    evaluation_count = 0
    evaluation_history = []
    return evaluation_count

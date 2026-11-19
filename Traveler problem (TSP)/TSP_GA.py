"""
Genetic Algorithm applied to the Travelling Salesman Problem (TSP).
 
Unlike the sphere/Rastrigin GAs, here each individual IS a permutation
of city indices (a tour), since that's the natural representation for
TSP -- no encoding/decoding needed.
 
    - Selection:  tournament selection
    - Crossover:  Order Crossover (OX) -- preserves relative order of
                  a sub-segment from one parent, fills in the rest from
                  the other parent, guaranteeing a valid permutation
    - Mutation:   swap mutation -- randomly swap two cities in the tour
    - Elitism:    best tour always survives to the next generation


Genetic Algorithm (GA) — the general algorithm loop
Initialization — generate a random population of candidate solutions.
Evaluation — compute the fitness of each individual using the objective function.
Selection — pick individuals to be "parents," biased toward higher fitness (common 
            methods: roulette wheel, tournament selection).
Crossover (recombination) — combine parts of two parents to produce offspring 
            (mimics reproduction/genetic mixing).
Mutation — randomly flip/alter small parts of an offspring to keep diversity and avoid 
            getting stuck.
Replacement — form the new generation (often keeping the best individual so far — "elitism").
Repeat steps 2–6 until a stopping condition (max generations, fitness good enough, 
            no improvement for a while).
"""


import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def init_population(pop_size, n_cities, seed=None):
    rng = np.random.RandomState(seed)
    base = np.arange(n_cities)
    return np.array([rng.permutation(base) for _ in range(pop_size)])


def evaluate(tour, dist_matrix):
    global evaluation_count
    evaluation_count += 1
    return tour_length(tour, dist_matrix)


def evaluate_population(population, dist_matrix):
    return np.array([evaluate(tour, dist_matrix) for tour in population])


def tournament_selection(population, fitness, k=3, rng=None):
    idx = rng.choice(len(population), size=k, replace=False)
    best_idx = idx[np.argmin(fitness[idx])]
    return population[best_idx].copy()


def order_crossover(parent1, parent2, rng=None):
    n = len(parent1)
    a, b = sorted(rng.choice(n, size=2, replace=False))

    child = np.full(n, -1, dtype=int)
    child[a:b] = parent1[a:b]

    fill_values = [c for c in parent2 if c not in child[a:b]]
    fill_idx = 0
    for i in list(range(0, a)) + list(range(b, n)):
        child[i] = fill_values[fill_idx]
        fill_idx += 1

    return child


def inversion_mutation(tour, mutation_rate, rng=None):
    mutated = tour.copy()
    if rng.rand() < mutation_rate:
        i, j = sorted(rng.choice(len(mutated), size=2, replace=False))
        mutated[i:j] = mutated[i:j][::-1]
    return mutated



def run_ga(
    dist_matrix,
    n_cities,
    pop_size=150,
    n_generations=500,
    tournament_k=3,
    crossover_rate=0.9,
    mutation_rate=0.2,
    elitism=True,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)

    population = init_population(pop_size, n_cities, seed=seed)
    fitness = evaluate_population(population, dist_matrix)

    best_history = []

    for gen in range(n_generations):
        new_population = []

        if elitism:
            best_idx = np.argmin(fitness)
            new_population.append(population[best_idx].copy())

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness, tournament_k, rng)
            parent2 = tournament_selection(population, fitness, tournament_k, rng)

            if rng.rand() < crossover_rate:
                child = order_crossover(parent1, parent2, rng)
            else:
                child = parent1.copy()

            child = inversion_mutation(child, mutation_rate, rng)
            new_population.append(child)

        population = np.array(new_population)
        fitness = evaluate_population(population, dist_matrix)

        best_history.append(np.min(fitness))
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 50 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:4d} | Best tour length: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    run_ga.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history



if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, history = run_ga(
        dmat, N_CITIES, pop_size=150, n_generations=500, seed=42,
    )

    print("\n=== Result (GA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best tour length")
    axes[0].set_title("GA Convergence on TSP")
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()

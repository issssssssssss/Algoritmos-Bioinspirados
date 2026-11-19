import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def init_population(pop_size, n_dim, bounds):
    lo, hi = bounds
    return np.random.uniform(lo, hi, size=(pop_size, n_dim))


def evaluate(ind):
    global evaluation_count
    evaluation_count += 1
    return rastrigin(ind)


def evaluate_population(population):
    return np.array([evaluate(ind) for ind in population])


def tournament_selection(population, fitness, k=3):
    idx = np.random.choice(len(population), size=k, replace=False)
    best_idx = idx[np.argmin(fitness[idx])]
    return population[best_idx].copy()


def blx_alpha_crossover(parent1, parent2, alpha=0.5, bounds=(-5.12, 5.12)):
    lo, hi = bounds
    child1, child2 = np.empty_like(parent1), np.empty_like(parent2)
    for i in range(len(parent1)):
        p_min, p_max = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
        diff = p_max - p_min
        low = max(lo, p_min - alpha * diff)
        high = min(hi, p_max + alpha * diff)
        child1[i] = np.random.uniform(low, high)
        child2[i] = np.random.uniform(low, high)
    return child1, child2


def gaussian_mutation(individual, mutation_rate, sigma, bounds=(-5.12, 5.12)):
    lo, hi = bounds
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.rand() < mutation_rate:
            mutated[i] += np.random.normal(0, sigma)
            mutated[i] = np.clip(mutated[i], lo, hi)
    return mutated


def run_ga(
    n_dim=10,
    pop_size=100,
    n_generations=200,
    bounds=(-5.12, 5.12),
    tournament_k=3,
    crossover_rate=0.8,
    mutation_rate=0.15,
    mutation_sigma=0.3,
    elitism=True,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    if seed is not None:
        np.random.seed(seed)

    population = init_population(pop_size, n_dim, bounds)
    fitness = evaluate_population(population)

    best_history = []
    mean_history = []
    best_position_history = []

    for gen in range(n_generations):
        new_population = []

        if elitism:
            best_idx = np.argmin(fitness)
            new_population.append(population[best_idx].copy())

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness, tournament_k)
            parent2 = tournament_selection(population, fitness, tournament_k)

            if np.random.rand() < crossover_rate:
                child1, child2 = blx_alpha_crossover(parent1, parent2, bounds=bounds)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            child1 = gaussian_mutation(child1, mutation_rate, mutation_sigma, bounds)
            child2 = gaussian_mutation(child2, mutation_rate, mutation_sigma, bounds)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = np.array(new_population)
        fitness = evaluate_population(population)

        best_idx = np.argmin(fitness)

        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))

        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:4d} | Best: {best_history[-1]:.6f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_ga.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_ga(
        n_dim=10,
        pop_size=100,
        n_generations=300,
        bounds=(-5.12, 5.12),
        seed=42,
    )

    print("\n=== Result (GA on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (log scale)")
    plt.title("GA Convergence on Rastrigin Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

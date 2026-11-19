import numpy as np
import matplotlib.pyplot as plt

from VRPTW_Problem import generate_instance, evaluate_solution, plot_routes, get_solution_details


evaluation_count = 0


def init_population(pop_size, n_customers, seed=None):
    rng = np.random.RandomState(seed)
    base = np.arange(1, n_customers + 1)
    return np.array([rng.permutation(base) for _ in range(pop_size)])


def evaluate(solution, instance):
    global evaluation_count
    evaluation_count += 1
    return evaluate_solution(solution, instance)


def evaluate_population(population, instance):
    return np.array([evaluate(sol, instance) for sol in population])


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


def inversion_mutation(solution, mutation_rate, rng=None):
    mutated = solution.copy()
    if rng.rand() < mutation_rate:
        i, j = sorted(rng.choice(len(mutated), size=2, replace=False))
        mutated[i:j] = mutated[i:j][::-1]
    return mutated


def run_ga(
    instance,
    n_customers,
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

    population = init_population(pop_size, n_customers, seed=seed)
    fitness = evaluate_population(population, instance)

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
        fitness = evaluate_population(population, instance)

        best_history.append(np.min(fitness))
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 50 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:4d} | Best cost: {best_history[-1]:.3f}")

    best_idx = np.argmin(fitness)
    run_ga.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history = run_ga(
        instance, N_CUSTOMERS, pop_size=150, n_generations=500, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (GA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best cost")
    axes[0].set_title("GA Convergence on VRPTW")
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

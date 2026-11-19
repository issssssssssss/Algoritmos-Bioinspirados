import numpy as np
import matplotlib.pyplot as plt

from VRPTW_Problem import (
    generate_instance,
    evaluate_solution,
    plot_routes,
    get_solution_details,
    random_key_to_permutation,
)


evaluation_count = 0


def evaluate(keys, instance):
    global evaluation_count
    evaluation_count += 1
    permutation = random_key_to_permutation(keys)
    return evaluate_solution(permutation, instance)


def run_iwo(
    instance,
    n_customers,
    initial_pop_size=10,
    max_pop_size=30,
    n_iterations=1000,
    min_seeds=0,
    max_seeds=5,
    initial_sigma=0.5,
    final_sigma=0.01,
    modulation_index=3,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Invasive Weed Optimization. Cada mala hierba produce un número de
    semillas que interpola linealmente entre min_seeds y max_seeds según
    qué tan buena es su aptitud respecto al resto de la población (las
    mejores producen más semillas). Las semillas se dispersan alrededor
    de la planta madre con desviación estándar `sigma`, que decrece de
    forma no lineal (exponente `modulation_index`) desde initial_sigma
    hasta final_sigma conforme avanzan las iteraciones. Al final de cada
    iteración, la población (plantas + semillas) se trunca a
    max_pop_size quedándose con las mejores.
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    population = rng.uniform(key_low, key_high, size=(initial_pop_size, dim))
    fitness = np.array([evaluate(population[i], instance) for i in range(initial_pop_size)])

    best_idx = np.argmin(fitness)
    global_best_pos = population[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        sigma = (
            ((n_iterations - it) / n_iterations) ** modulation_index
        ) * (initial_sigma - final_sigma) + final_sigma

        worst_fit = fitness.max()
        best_fit = fitness.min()

        seeds = []
        for i in range(len(population)):
            if worst_fit == best_fit:
                n_seeds = max_seeds
            else:
                ratio = (worst_fit - fitness[i]) / (worst_fit - best_fit)
                n_seeds = int(round(min_seeds + ratio * (max_seeds - min_seeds)))

            for _ in range(n_seeds):
                offspring = population[i] + sigma * rng.normal(0, 1, size=dim)
                offspring = np.clip(offspring, key_low, key_high)
                seeds.append(offspring)

        if seeds:
            seeds = np.array(seeds)
            seed_fitness = np.array([evaluate(seeds[k], instance) for k in range(len(seeds))])
            population = np.vstack([population, seeds])
            fitness = np.concatenate([fitness, seed_fitness])

        if len(population) > max_pop_size:
            order = np.argsort(fitness)[:max_pop_size]
            population = population[order]
            fitness = fitness[order]

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = population[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f} | Pop size: {len(population)}")

    run_iwo.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_iwo(
        instance, N_CUSTOMERS, initial_pop_size=10, max_pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (IWO on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("IWO Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

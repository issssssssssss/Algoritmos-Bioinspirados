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


def run_bfoa(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    step_size=0.1,
    reproduction_period=20,
    elimination_period=100,
    elimination_prob=0.1,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Bacterial Foraging Optimization Algorithm.
    Cada iteración es un paso de quimiotaxis: cada bacteria intenta una
    "tumbación" en una dirección aleatoria y se mueve `step_size` en esa
    dirección; el movimiento se acepta (equivalente a "nadar") si mejora
    su costo, y se descarta si no. Cada `reproduction_period`
    iteraciones, la mitad menos apta de la población es reemplazada por
    copias de la mitad más apta (reproducción). Cada `elimination_period`
    iteraciones, cada bacteria tiene probabilidad `elimination_prob` de
    ser dispersada a una posición aleatoria (eliminación-dispersión).
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    population = rng.uniform(key_low, key_high, size=(pop_size, dim))
    fitness = np.array([evaluate(population[i], instance) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    global_best_pos = population[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        # --- Quimiotaxis (tumble + accept-if-better) ---
        for i in range(pop_size):
            direction = rng.uniform(-1, 1, size=dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm

            candidate = population[i] + step_size * direction
            candidate = np.clip(candidate, key_low, key_high)
            cand_fit = evaluate(candidate, instance)

            if cand_fit < fitness[i]:
                population[i] = candidate
                fitness[i] = cand_fit

        # --- Reproducción periódica ---
        if (it + 1) % reproduction_period == 0:
            order = np.argsort(fitness)
            half = pop_size // 2
            best_half_idx = order[:half]
            worst_half_idx = order[half: 2 * half]
            for src, dst in zip(best_half_idx, worst_half_idx):
                population[dst] = population[src].copy()
                fitness[dst] = fitness[src]

        # --- Eliminación-dispersión periódica ---
        if (it + 1) % elimination_period == 0:
            for i in range(pop_size):
                if rng.rand() < elimination_prob:
                    population[i] = rng.uniform(key_low, key_high, size=dim)
                    fitness[i] = evaluate(population[i], instance)

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = population[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_bfoa.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_bfoa(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (BFOA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("BFOA Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

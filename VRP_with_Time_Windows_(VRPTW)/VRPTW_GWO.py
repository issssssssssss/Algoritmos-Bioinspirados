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


def run_gwo(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    wolves = rng.uniform(key_low, key_high, size=(pop_size, dim))
    fitness = np.array([evaluate(wolves[i], instance) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    global_best_pos = wolves[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        order = np.argsort(fitness)
        alpha_pos = wolves[order[0]]
        beta_pos = wolves[order[1]] if pop_size > 1 else wolves[order[0]]
        delta_pos = wolves[order[2]] if pop_size > 2 else wolves[order[0]]

        a = 2 - it * (2.0 / n_iterations)

        new_wolves = np.zeros_like(wolves)
        for i in range(pop_size):
            X = np.zeros((3, dim))
            for k, leader in enumerate([alpha_pos, beta_pos, delta_pos]):
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - wolves[i])
                X[k] = leader - A * D
            new_wolves[i] = X.mean(axis=0)

        wolves = np.clip(new_wolves, key_low, key_high)
        fitness = np.array([evaluate(wolves[i], instance) for i in range(pop_size)])

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = wolves[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_gwo.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_gwo(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (GWO on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("GWO Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

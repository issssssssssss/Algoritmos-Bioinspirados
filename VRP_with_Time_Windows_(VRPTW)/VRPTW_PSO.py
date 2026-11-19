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


def run_pso(
    instance,
    n_customers,
    n_particles=80,
    n_iterations=500,
    w=0.7,
    c1=1.5,
    c2=1.5,
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

    position = rng.uniform(key_low, key_high, size=(n_particles, dim))
    velocity = np.zeros((n_particles, dim))

    fitness = np.array([evaluate(position[i], instance) for i in range(n_particles)])

    personal_best_pos = position.copy()
    personal_best_fit = fitness.copy()

    global_best_idx = np.argmin(personal_best_fit)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = personal_best_fit[global_best_idx]

    best_history = []

    for it in range(n_iterations):
        r1 = rng.rand(n_particles, dim)
        r2 = rng.rand(n_particles, dim)

        velocity = (
            w * velocity
            + c1 * r1 * (personal_best_pos - position)
            + c2 * r2 * (global_best_pos - position)
        )
        position = position + velocity
        position = np.clip(position, key_low, key_high)

        fitness = np.array([evaluate(position[i], instance) for i in range(n_particles)])

        improved = fitness < personal_best_fit
        personal_best_pos[improved] = position[improved]
        personal_best_fit[improved] = fitness[improved]

        current_best_idx = np.argmin(personal_best_fit)
        if personal_best_fit[current_best_idx] < global_best_fit:
            global_best_fit = personal_best_fit[current_best_idx]
            global_best_pos = personal_best_pos[current_best_idx].copy()

        best_history.append(global_best_fit)
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_pso.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history = run_pso(
        instance, N_CUSTOMERS, n_particles=80, n_iterations=500, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (PSO on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Best cost")
    axes[0].set_title("PSO Convergence on VRPTW")
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

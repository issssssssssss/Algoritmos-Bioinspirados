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


def run_fa(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    alpha=0.2,
    alpha_decay=0.97,
    beta0=1.0,
    gamma=1.0,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Firefly Algorithm. Cada luciérnaga i se mueve hacia toda luciérnaga j
    más brillante (de menor costo), con una atracción que decae con el
    cuadrado de la distancia (parámetro gamma) más un término aleatorio
    que se reduce geométricamente con las iteraciones (alpha * alpha_decay^it).
    Las comparaciones dentro de una iteración usan el fitness al inicio de
    la iteración (actualización por lotes), y se reevalúa una vez por
    luciérnaga al final de cada iteración.
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    fireflies = rng.uniform(key_low, key_high, size=(pop_size, dim))
    fitness = np.array([evaluate(fireflies[i], instance) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    global_best_pos = fireflies[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        alpha_it = alpha * (alpha_decay ** it)
        new_fireflies = fireflies.copy()

        for i in range(pop_size):
            moved = False
            for j in range(pop_size):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    rand_term = alpha_it * (rng.rand(dim) - 0.5)
                    new_fireflies[i] = new_fireflies[i] + beta * (fireflies[j] - new_fireflies[i]) + rand_term
                    moved = True

            if not moved:
                # la luciérnaga más brillante de todas: paso aleatorio puro
                new_fireflies[i] = new_fireflies[i] + alpha_it * (rng.rand(dim) - 0.5)

        fireflies = np.clip(new_fireflies, key_low, key_high)
        fitness = np.array([evaluate(fireflies[i], instance) for i in range(pop_size)])

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = fireflies[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_fa.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_fa(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (FA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("FA Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

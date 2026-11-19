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


def run_acor(
    instance,
    n_customers,
    archive_size=30,
    n_ants=30,
    n_iterations=1000,
    q=0.5,
    xi=0.85,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    ACOR (Ant Colony Optimization for continuous domains, Socha & Dorigo 2008).
    Mantiene un archivo de `archive_size` soluciones ordenadas por calidad;
    cada dimensión se muestrea de una mezcla de gaussianas centradas en las
    soluciones del archivo, con pesos que favorecen a las mejores (parámetro q)
    y una desviación proporcional a la dispersión del archivo (parámetro xi).
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    archive = rng.uniform(key_low, key_high, size=(archive_size, dim))
    archive_fit = np.array([evaluate(archive[i], instance) for i in range(archive_size)])

    order = np.argsort(archive_fit)
    archive = archive[order]
    archive_fit = archive_fit[order]

    ranks = np.arange(1, archive_size + 1)
    weights = (1.0 / (q * archive_size * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((ranks - 1) / (q * archive_size)) ** 2
    )
    weights = weights / weights.sum()

    global_best_pos = archive[0].copy()
    global_best_fit = archive_fit[0]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        new_solutions = np.zeros((n_ants, dim))

        for a in range(n_ants):
            for d in range(dim):
                chosen = rng.choice(archive_size, p=weights)
                mean = archive[chosen, d]
                sigma = xi * np.mean(np.abs(archive[:, d] - mean)) + 1e-6
                new_solutions[a, d] = rng.normal(mean, sigma)

        new_solutions = np.clip(new_solutions, key_low, key_high)
        new_fit = np.array([evaluate(new_solutions[a], instance) for a in range(n_ants)])

        combined_pos = np.vstack([archive, new_solutions])
        combined_fit = np.concatenate([archive_fit, new_fit])

        best_order = np.argsort(combined_fit)[:archive_size]
        archive = combined_pos[best_order]
        archive_fit = combined_fit[best_order]

        if archive_fit[0] < global_best_fit:
            global_best_fit = archive_fit[0]
            global_best_pos = archive[0].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(archive_fit))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_acor.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_acor(
        instance, N_CUSTOMERS, archive_size=30, n_ants=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (ACOR on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("ACOR Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

from VRPTW_Problem import generate_instance, evaluate_solution, plot_routes, get_solution_details


evaluation_count = 0


def evaluate(solution, instance):
    global evaluation_count
    evaluation_count += 1
    return evaluate_solution(solution, instance)


def construct_tour(pheromone, heuristic, n_customers, alpha, beta, rng):
    """Construye un giant tour (permutación de clientes 1..n_customers)
    guiado por feromona y heurística 1/distancia, empezando en el
    depósito (índice 0)."""
    unvisited = list(range(1, n_customers + 1))
    current = 0
    tour = []

    for _ in range(n_customers):
        weights = np.array([
            (pheromone[current, j] ** alpha) * (heuristic[current, j] ** beta)
            for j in unvisited
        ])
        total = weights.sum()

        if total <= 0 or not np.isfinite(total):
            probs = np.ones(len(unvisited)) / len(unvisited)
        else:
            probs = weights / total

        next_idx = rng.choice(len(unvisited), p=probs)
        next_city = unvisited.pop(next_idx)
        tour.append(next_city)
        current = next_city

    return np.array(tour)


def run_aco(
    instance,
    n_customers,
    n_ants=30,
    n_iterations=1000,
    alpha=1.0,
    beta=3.0,
    rho=0.3,
    Q=100.0,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)

    n_nodes = n_customers + 1  # depósito + clientes
    dmat = instance.dist_matrix
    with np.errstate(divide="ignore"):
        heuristic = np.where(dmat > 0, 1.0 / dmat, 0.0)

    pheromone = np.ones((n_nodes, n_nodes))

    global_best_solution = None
    global_best_fit = np.inf

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        tours = [construct_tour(pheromone, heuristic, n_customers, alpha, beta, rng)
                  for _ in range(n_ants)]
        fitness = np.array([evaluate(tour, instance) for tour in tours])

        pheromone *= (1 - rho)
        for tour, fit in zip(tours, fitness):
            deposit = Q / (fit + 1e-9)
            path = [0] + list(tour) + [0]
            for a, b in zip(path[:-1], path[1:]):
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_solution = tours[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_solution.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_aco.evaluation_history = evaluation_history
    return global_best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_aco(
        instance, N_CUSTOMERS, n_ants=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (ACO on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("ACO Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

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


def run_afsa(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    visual=0.3,
    step=0.1,
    try_number=5,
    crowd_factor=0.8,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Artificial Fish Swarm Algorithm. Cada pez intenta, en orden de
    prioridad: (1) enjambrarse (swarm) hacia el centro de sus vecinos
    dentro del radio `visual` si ese centro es mejor y la zona no está
    saturada; (2) seguir (follow) al mejor vecino visible si es mejor
    que él; (3) si ninguna de las anteriores mejora su posición, cazar
    (prey) probando `try_number` puntos aleatorios dentro de su campo
    visual y quedándose con el mejor, o dar un paso aleatorio si
    ninguno mejora.
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers

    positions = rng.uniform(key_low, key_high, size=(pop_size, dim))
    fitness = np.array([evaluate(positions[i], instance) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    global_best_pos = positions[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        new_positions = positions.copy()
        new_fitness = fitness.copy()

        for i in range(pop_size):
            xi = positions[i]
            fi = fitness[i]

            dists = np.linalg.norm(positions - xi, axis=1)
            neighbor_idx = np.where((dists <= visual) & (dists > 0))[0]

            moved = False

            # --- Swarm ---
            if len(neighbor_idx) > 0:
                crowded = len(neighbor_idx) > crowd_factor * pop_size
                if not crowded and fitness[neighbor_idx].mean() < fi:
                    center = positions[neighbor_idx].mean(axis=0)
                    direction = center - xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        candidate = xi + step * direction / norm * rng.rand()
                        candidate = np.clip(candidate, key_low, key_high)
                        cand_fit = evaluate(candidate, instance)
                        if cand_fit < fi:
                            new_positions[i] = candidate
                            new_fitness[i] = cand_fit
                            moved = True

            # --- Follow ---
            if not moved and len(neighbor_idx) > 0:
                best_neighbor = neighbor_idx[np.argmin(fitness[neighbor_idx])]
                if fitness[best_neighbor] < fi:
                    direction = positions[best_neighbor] - xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        candidate = xi + step * direction / norm * rng.rand()
                        candidate = np.clip(candidate, key_low, key_high)
                        cand_fit = evaluate(candidate, instance)
                        if cand_fit < fi:
                            new_positions[i] = candidate
                            new_fitness[i] = cand_fit
                            moved = True

            # --- Prey ---
            if not moved:
                best_try_pos = None
                best_try_fit = fi
                for _ in range(try_number):
                    direction = rng.uniform(-1, 1, size=dim)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-12:
                        continue
                    candidate = xi + visual * direction / norm * rng.rand()
                    candidate = np.clip(candidate, key_low, key_high)
                    cand_fit = evaluate(candidate, instance)
                    if cand_fit < best_try_fit:
                        best_try_fit = cand_fit
                        best_try_pos = candidate

                if best_try_pos is not None:
                    new_positions[i] = best_try_pos
                    new_fitness[i] = best_try_fit
                else:
                    candidate = xi + step * rng.uniform(-1, 1, size=dim)
                    candidate = np.clip(candidate, key_low, key_high)
                    cand_fit = evaluate(candidate, instance)
                    new_positions[i] = candidate
                    new_fitness[i] = cand_fit

        positions = new_positions
        fitness = new_fitness

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = positions[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_afsa.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_afsa(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (AFSA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("AFSA Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

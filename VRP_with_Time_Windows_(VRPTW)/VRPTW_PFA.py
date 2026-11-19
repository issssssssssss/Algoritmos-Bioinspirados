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


def run_pfa(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    selection_rate=0.5,
    max_seeds=5,
    sigma=0.3,
    pollination_rate=0.2,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Paddy Field Algorithm. Cada iteración tiene tres fases:
      1. Selección: se quedan las `selection_rate` * pop_size mejores
         plantas (paddies).
      2. Siembra (seeding): cada planta seleccionada dispersa semillas
         cerca de sí misma, en cantidad proporcional a su rango de
         aptitud (las mejores producen hasta max_seeds semillas, con
         desviación `sigma`) -- igual que la dispersión en IWO.
      3. Polinización: se generan soluciones adicionales combinando
         (mezcla convexa aleatoria) pares de plantas seleccionadas.
    Semillas + polinizados compiten con la población actual y se
    conservan las pop_size mejores (selección por truncamiento).
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
        order = np.argsort(fitness)
        n_selected = max(2, int(round(selection_rate * pop_size)))
        selected_idx = order[:n_selected]

        worst_fit = fitness[order[-1]]
        best_fit = fitness[order[0]]

        # --- Siembra ---
        seeds = []
        for idx in selected_idx:
            if worst_fit == best_fit:
                n_seeds = max_seeds
            else:
                ratio = (worst_fit - fitness[idx]) / (worst_fit - best_fit)
                n_seeds = max(1, int(round(ratio * max_seeds)))
            for _ in range(n_seeds):
                offspring = population[idx] + sigma * rng.normal(0, 1, size=dim)
                offspring = np.clip(offspring, key_low, key_high)
                seeds.append(offspring)

        # --- Polinización ---
        n_pollinate = max(1, int(round(pollination_rate * pop_size)))
        pollinated = []
        for _ in range(n_pollinate):
            a, b = rng.choice(selected_idx, size=2, replace=False)
            w = rng.rand()
            child = w * population[a] + (1 - w) * population[b]
            pollinated.append(child)

        offspring = np.array(seeds + pollinated)
        offspring_fitness = np.array([evaluate(offspring[k], instance) for k in range(len(offspring))])

        combined_pos = np.vstack([population, offspring])
        combined_fit = np.concatenate([fitness, offspring_fitness])

        keep_idx = np.argsort(combined_fit)[:pop_size]
        population = combined_pos[keep_idx]
        fitness = combined_fit[keep_idx]

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

    run_pfa.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_pfa(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (PFA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("PFA Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

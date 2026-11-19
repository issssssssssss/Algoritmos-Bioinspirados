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


def run_ais(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    n_select=10,
    clone_factor=5,
    beta=1.0,
    replace_rate=0.2,
    key_low=0.0,
    key_high=1.0,
    seed=None,
    verbose=True,
):
    """
    Artificial Immune System (selección clonal, estilo CLONALG).
    En cada generación:
      1. Se seleccionan los `n_select` anticuerpos de mayor afinidad
         (menor costo).
      2. Cada uno produce clones (más clones cuanto mejor su afinidad)
         que se hipermutan con una tasa inversamente proporcional a su
         rango (los mejores mutan menos: explotación local fina).
      3. Cada anticuerpo original es reemplazado por su mejor clon si
         lo mejora.
      4. Se reemplaza una fracción `replace_rate` de los peores
         anticuerpos por otros aleatorios nuevos (introducción de
         diversidad).
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

    n_select = min(n_select, pop_size)

    for it in range(n_iterations):
        order = np.argsort(fitness)
        selected_idx = order[:n_select]

        clones = []
        clone_counts = []
        for rank, idx in enumerate(selected_idx):
            n_clones = max(1, int(round(clone_factor * (n_select - rank) / n_select)))
            clone_counts.append(n_clones)
            mutation_rate = beta * np.exp(-(rank + 1))
            for _ in range(n_clones):
                clone = population[idx] + rng.normal(0, mutation_rate, size=dim)
                clone = np.clip(clone, key_low, key_high)
                clones.append(clone)

        clones = np.array(clones)
        clone_fitness = np.array([evaluate(clones[k], instance) for k in range(len(clones))])

        ptr = 0
        for rank, idx in enumerate(selected_idx):
            n_clones = clone_counts[rank]
            local_fit = clone_fitness[ptr: ptr + n_clones]
            local_pos = clones[ptr: ptr + n_clones]
            ptr += n_clones

            best_clone = np.argmin(local_fit)
            if local_fit[best_clone] < fitness[idx]:
                population[idx] = local_pos[best_clone]
                fitness[idx] = local_fit[best_clone]

        n_replace = max(1, int(round(replace_rate * pop_size)))
        worst_idx = np.argsort(fitness)[-n_replace:]
        for idx in worst_idx:
            population[idx] = rng.uniform(key_low, key_high, size=dim)
            fitness[idx] = evaluate(population[idx], instance)

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

    run_ais.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_ais(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (AIS on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("AIS Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

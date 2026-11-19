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


def bits_to_keys(bits, n_customers, bits_per_key):
    """Convierte un vector binario (n_customers * bits_per_key) en un
    vector de claves reales en [0, 1], interpretando cada bloque de
    bits_per_key bits como un entero normalizado (random-key)."""
    bits = bits.reshape(n_customers, bits_per_key)
    powers = 2 ** np.arange(bits_per_key - 1, -1, -1)
    integers = bits @ powers
    max_int = 2 ** bits_per_key - 1
    return integers / max_int


def evaluate(bits, instance, n_customers, bits_per_key):
    global evaluation_count
    evaluation_count += 1
    keys = bits_to_keys(bits, n_customers, bits_per_key)
    permutation = random_key_to_permutation(keys)
    return evaluate_solution(permutation, instance)


def run_cga(
    instance,
    n_customers,
    bits_per_key=16,
    pop_size=100,
    n_generations=500,
    seed=None,
    verbose=True,
):
    """
    Compact Genetic Algorithm (cGA). En vez de una población explícita,
    mantiene un vector de probabilidad (uno por bit) que simula una
    población virtual de tamaño `pop_size`. Cada generación:
      1. Se muestrean dos individuos binarios A y B a partir del vector
         de probabilidad (cada bit ~ Bernoulli(prob_vector[i])).
      2. Se decodifican a random-keys -> permutación -> costo VRPTW
         (misma decodificación que PSO/GWO/WOA/ABC/ACOR).
      3. El ganador (menor costo) empuja el vector de probabilidad hacia
         sus propios bits en los loci donde A y B difieren, con un paso
         de 1/pop_size -- así `pop_size` controla qué tan rápido converge
         el vector, igual que el tamaño de población en un GA clásico.
      4. Se recorta el vector de probabilidad a [1/pop_size, 1-1/pop_size]
         para evitar que se sature por completo en 0 o 1 y pierda toda
         capacidad de explorar.
    Cada generación cuesta exactamente 2 evaluaciones (A y B) -- de ahí
    que en el script de benchmark las "generaciones" de cGA se calculen
    como EVAL_BUDGET // 2.
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    dim = n_customers * bits_per_key

    prob_vector = np.full(dim, 0.5)

    best_bits = None
    best_fitness = np.inf

    best_history = []

    lower_bound = 1.0 / pop_size
    upper_bound = 1.0 - lower_bound

    for gen in range(n_generations):
        a = (rng.rand(dim) < prob_vector).astype(int)
        b = (rng.rand(dim) < prob_vector).astype(int)

        fitness_a = evaluate(a, instance, n_customers, bits_per_key)
        fitness_b = evaluate(b, instance, n_customers, bits_per_key)

        if fitness_a <= fitness_b:
            winner, winner_fit = a, fitness_a
            loser = b
        else:
            winner, winner_fit = b, fitness_b
            loser = a

        diff = winner != loser
        prob_vector[diff & (winner == 1)] += 1.0 / pop_size
        prob_vector[diff & (winner == 0)] -= 1.0 / pop_size
        prob_vector = np.clip(prob_vector, lower_bound, upper_bound)

        if winner_fit < best_fitness:
            best_fitness = winner_fit
            best_bits = winner.copy()

        best_history.append(best_fitness)
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 50 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:4d} | Best cost: {best_fitness:.3f}")

    run_cga.evaluation_history = evaluation_history
    best_keys = bits_to_keys(best_bits, n_customers, bits_per_key)
    best_solution = random_key_to_permutation(best_keys)
    return best_solution, best_fitness, best_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history = run_cga(
        instance, N_CUSTOMERS, bits_per_key=16, pop_size=100, n_generations=500, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (cGA on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best cost")
    axes[0].set_title("cGA Convergence on VRPTW")
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

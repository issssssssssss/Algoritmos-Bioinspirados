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


def _fitness_to_prob(fit):
    adjusted = 1.0 / (1.0 + fit - fit.min())
    return adjusted / adjusted.sum()


def run_abc(
    instance,
    n_customers,
    pop_size=30,
    n_iterations=1000,
    limit=50,
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

    food_sources = rng.uniform(key_low, key_high, size=(pop_size, dim))
    fitness = np.array([evaluate(food_sources[i], instance) for i in range(pop_size)])
    trial_counters = np.zeros(pop_size, dtype=int)

    best_idx = np.argmin(fitness)
    global_best_pos = food_sources[best_idx].copy()
    global_best_fit = fitness[best_idx]

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        # --- Employed bees ---
        for i in range(pop_size):
            k = rng.choice([j for j in range(pop_size) if j != i])
            phi = rng.uniform(-1, 1, size=dim)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, key_low, key_high)
            candidate_fit = evaluate(candidate, instance)

            if candidate_fit < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fit
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1

        # --- Onlooker bees ---
        probs = _fitness_to_prob(fitness)
        for _ in range(pop_size):
            i = rng.choice(pop_size, p=probs)
            k = rng.choice([j for j in range(pop_size) if j != i])
            phi = rng.uniform(-1, 1, size=dim)
            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, key_low, key_high)
            candidate_fit = evaluate(candidate, instance)

            if candidate_fit < fitness[i]:
                food_sources[i] = candidate
                fitness[i] = candidate_fit
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1

        # --- Scout bees ---
        for i in range(pop_size):
            if trial_counters[i] > limit:
                food_sources[i] = rng.uniform(key_low, key_high, size=dim)
                fitness[i] = evaluate(food_sources[i], instance)
                trial_counters[i] = 0

        it_best_idx = np.argmin(fitness)
        if fitness[it_best_idx] < global_best_fit:
            global_best_fit = fitness[it_best_idx]
            global_best_pos = food_sources[it_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_pos.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_abc.evaluation_history = evaluation_history
    best_solution = random_key_to_permutation(global_best_pos)
    return best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_abc(
        instance, N_CUSTOMERS, pop_size=30, n_iterations=1000, limit=50, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (ABC on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("ABC Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

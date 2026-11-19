import numpy as np
import knapsack_common as common


def _binary_decode(x):
    return np.asarray(x, dtype=int)


def run_ga(
    n_dim=10, pop_size=100, n_generations=200, bounds=(0, 1),
    tournament_k=3, crossover_rate=0.8, mutation_rate=0.15,
    mutation_sigma=0.3, elitism=True, seed=None, verbose=True,
    items=None, capacity=None,
):
    common.reset()
    rng = np.random.RandomState(seed)
    population = rng.randint(0, 2, size=(pop_size, n_dim)).astype(int)
    fitness = common.evaluate_population(population, items, capacity, decode_fn=_binary_decode)
    best_history, mean_history, best_position_history = [], [], []
    best_idx = np.argmin(fitness)
    best_ever = population[best_idx].copy()
    best_ever_f = fitness[best_idx]

    def tournament():
        idx = rng.choice(pop_size, size=tournament_k, replace=False)
        return population[idx[np.argmin(fitness[idx])]].copy()

    for gen in range(n_generations):
        new_population = []
        if elitism:
            new_population.append(best_ever.copy())
        while len(new_population) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.rand() < crossover_rate:
                point = rng.randint(1, n_dim)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                mask = rng.rand(n_dim) < mutation_rate
                c[mask] = 1 - c[mask]
                new_population.append(c)
                if len(new_population) >= pop_size:
                    break
        population = np.asarray(new_population[:pop_size])
        fitness = common.evaluate_population(population, items, capacity, decode_fn=_binary_decode)
        idx = np.argmin(fitness)
        if fitness[idx] < best_ever_f:
            best_ever_f = fitness[idx]
            best_ever = population[idx].copy()
        best_history.append(best_ever_f)
        mean_history.append(np.mean(fitness))
        best_position_history.append(best_ever.copy())
        common.evaluation_history.append(common.evaluation_count)
        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:4d} | Best value: {-best_ever_f:.6f} | Mean objective: {mean_history[-1]:.6f}")
    run_ga.evaluation_history = common.evaluation_history
    return common.result_position(best_ever), best_ever_f, best_history, mean_history, best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_ga(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=100,
        n_generations=300,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (GA on Knapsack) ===")
    print("Best selection:", best_x)
    total_weight = sum(DEFAULT_ITEMS[i][0] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    total_value = sum(DEFAULT_ITEMS[i][1] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    print("Total weight:", total_weight)
    print("Total value:", total_value)
    print("Best value:", -best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.title("GA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

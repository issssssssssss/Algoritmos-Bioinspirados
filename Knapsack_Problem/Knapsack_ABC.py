import numpy as np
import knapsack_common as common


def _continuous_to_binary(x, threshold=0.5):
    return (np.asarray(x) >= threshold).astype(int)


def run_abc(
    n_dim=10, pop_size=30, n_iterations=300, bounds=(0, 1),
    limit=50, seed=None, verbose=True, items=None, capacity=None,
):
    common.reset()
    rng = np.random.RandomState(seed)
    lo, hi = bounds
    population = rng.uniform(lo, hi, (pop_size, n_dim))
    fitness = common.evaluate_population(population, items, capacity, decode_fn=_continuous_to_binary)
    trial = np.zeros(pop_size, dtype=int)
    best_history, mean_history, best_position_history = [], [], []
    best_idx = np.argmin(fitness)
    best_x, best_f = population[best_idx].copy(), fitness[best_idx]

    def neighbor(i):
        j = rng.randint(n_dim)
        k = rng.choice([q for q in range(pop_size) if q != i])
        phi = rng.uniform(-1, 1)
        v = population[i].copy()
        v[j] += phi * (population[i][j] - population[k][j])
        return np.clip(v, lo, hi)

    for it in range(n_iterations):
        for i in range(pop_size):
            v = neighbor(i); fv = common.evaluate(v, items, capacity, decode_fn=_continuous_to_binary)
            if fv < fitness[i]:
                population[i], fitness[i], trial[i] = v, fv, 0
            else: trial[i] += 1
        transformed = 1.0 / (1.0 + np.maximum(0, fitness - fitness.min()))
        probs = transformed / transformed.sum()
        for _ in range(pop_size):
            i = rng.choice(pop_size, p=probs)
            v = neighbor(i); fv = common.evaluate(v, items, capacity, decode_fn=_continuous_to_binary)
            if fv < fitness[i]:
                population[i], fitness[i], trial[i] = v, fv, 0
            else: trial[i] += 1
        for i in range(pop_size):
            if trial[i] > limit:
                population[i] = rng.uniform(lo, hi, n_dim)
                fitness[i] = common.evaluate(population[i], items, capacity, decode_fn=_continuous_to_binary)
                trial[i] = 0
        idx = np.argmin(fitness)
        if fitness[idx] < best_f:
            best_x, best_f = population[idx].copy(), fitness[idx]
        best_history.append(best_f); mean_history.append(np.mean(fitness))
        best_position_history.append(best_x.copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best value: {-best_f:.6f} | Mean objective: {mean_history[-1]:.6f}")
    run_abc.evaluation_history = common.evaluation_history
    return common.result_position(best_x), best_f, best_history, mean_history, best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_abc(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (ABC on Knapsack) ===")
    print("Best selection:", best_x)
    total_weight = sum(DEFAULT_ITEMS[i][0] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    total_value = sum(DEFAULT_ITEMS[i][1] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    print("Total weight:", total_weight)
    print("Total value:", total_value)
    print("Best value:", -best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("ABC convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

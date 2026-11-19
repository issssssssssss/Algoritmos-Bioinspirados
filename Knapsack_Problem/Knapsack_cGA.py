import numpy as np
import knapsack_common as common


def _binary_decode(x):
    return np.asarray(x, dtype=int)


def run_cga(
    n_dim=10, bits_per_var=1, bounds=(0, 1), pop_size=100,
    n_generations=8000, seed=None, verbose=True, items=None, capacity=None,
):
    common.reset()
    rng = np.random.RandomState(seed)
    prob_vector = np.full(n_dim, 0.5)
    best_ever = None
    best_ever_f = np.inf
    best_history, best_position_history = [], []
    step = 1.0 / pop_size

    for gen in range(n_generations):
        ind1 = (rng.rand(n_dim) < prob_vector).astype(int)
        ind2 = (rng.rand(n_dim) < prob_vector).astype(int)
        f1 = common.evaluate(ind1, items, capacity, decode_fn=_binary_decode)
        f2 = common.evaluate(ind2, items, capacity, decode_fn=_binary_decode)
        if f1 <= f2:
            winner, loser, f_win = ind1, ind2, f1
        else:
            winner, loser, f_win = ind2, ind1, f2
        if f_win < best_ever_f:
            best_ever_f = f_win
            best_ever = winner.copy()
        diff = winner != loser
        prob_vector[diff & (winner == 1)] += step
        prob_vector[diff & (winner == 0)] -= step
        prob_vector = np.clip(prob_vector, 0, 1)
        best_history.append(best_ever_f)
        best_position_history.append(best_ever.copy())
        common.evaluation_history.append(common.evaluation_count)
        if verbose and (gen % 300 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:5d} | Best value: {-best_ever_f:.6f}")
        if np.all((prob_vector == 0) | (prob_vector == 1)):
            if verbose:
                print(f"Probability vector converged at generation {gen}")
            break
    run_cga.evaluation_history = common.evaluation_history
    return common.result_position(best_ever), best_ever_f, best_history, best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, best_pos_hist = run_cga(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=100,
        n_generations=500,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (cGA on Knapsack) ===")
    print("Best selection:", best_x)
    total_weight = sum(DEFAULT_ITEMS[i][0] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    total_value = sum(DEFAULT_ITEMS[i][1] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    print("Total weight:", total_weight)
    print("Total value:", total_value)
    print("Best value:", -best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.title("cGA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

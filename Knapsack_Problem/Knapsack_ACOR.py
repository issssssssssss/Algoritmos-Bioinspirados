import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_acor(
    n_dim=10, archive_size=30, n_ants=30, n_iterations=300,
    bounds=(0, 1), q=0.5, xi=0.85, seed=None, verbose=True,
    items=None, capacity=None,
):
    common.reset()
    rng = np.random.RandomState(seed); lo, hi = bounds
    archive = rng.uniform(lo, hi, (archive_size, n_dim))
    archive_f = evaluate_population(archive, items, capacity)
    order = np.argsort(archive_f); archive, archive_f = archive[order], archive_f[order]
    ranks = np.arange(1, archive_size + 1)
    weights = np.exp(-(ranks - 1)**2/(2*q*q*archive_size*archive_size))
    probs = weights/weights.sum()
    best_history, mean_history, best_position_history = [], [], []
    for it in range(n_iterations):
        ants = np.empty((n_ants, n_dim))
        for a in range(n_ants):
            l = rng.choice(archive_size, p=probs)
            for j in range(n_dim):
                sigma = xi/(archive_size-1)*np.sum(np.abs(archive[:,j]-archive[l,j]))
                ants[a,j] = rng.normal(archive[l,j], sigma if sigma > 0 else 1e-6)
        ants = np.clip(ants, lo, hi)
        ant_f = evaluate_population(ants, items, capacity)
        combined = np.vstack((archive, ants)); combined_f = np.concatenate((archive_f, ant_f))
        order = np.argsort(combined_f)[:archive_size]
        archive, archive_f = combined[order], combined_f[order]
        best_history.append(archive_f[0]); mean_history.append(np.mean(combined_f))
        best_position_history.append(archive[0].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best value: {-archive_f[0]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    run_acor.evaluation_history = common.evaluation_history
    return result_position(archive[0]), archive_f[0], best_history, mean_history, best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_acor(
        n_dim=len(DEFAULT_ITEMS),
        archive_size=30,
        n_ants=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (ACOR on Knapsack) ===")
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
    plt.title("ACOR convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

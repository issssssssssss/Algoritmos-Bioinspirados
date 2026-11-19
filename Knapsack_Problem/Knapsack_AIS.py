import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_ais(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),
    n_select=10,clones_per_antibody=5,beta=2.0,replace_rate=0.2,
    seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]; n_replace=max(1,int(replace_rate*pop_size))
    for it in range(n_iterations):
        order=np.argsort(fitness); selected=order[:min(n_select,pop_size)]
        for rank,idx in enumerate(selected):
            affinity=rank/max(1,len(selected)-1)
            mutation_std=(.01+.5*affinity)*(hi-lo)
            bx=population[idx].copy(); bf=fitness[idx]
            for _ in range(clones_per_antibody):
                clone=np.clip(population[idx]+rng.normal(0,mutation_std,n_dim),lo,hi)
                cf=evaluate(clone,items,capacity)
                if cf<bf: bx,bf=clone,cf
            population[idx],fitness[idx]=bx,bf
        for idx in np.argsort(fitness)[-n_replace:]:
            population[idx]=rng.uniform(lo,hi,n_dim); fitness[idx]=evaluate(population[idx],items,capacity)
        best_idx=np.argmin(fitness); best_history.append(fitness[best_idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[best_idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[best_idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    idx=np.argmin(fitness); run_ais.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_ais(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (AIS on Knapsack) ===")
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
    plt.title("AIS convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_iwo(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),min_seeds=0,max_seeds=5,exponent=2,sigma_initial=.5,sigma_final=.001,
    seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds; pop_max=pop_size
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]
    for it in range(n_iterations):
        sigma=((n_iterations-it)/n_iterations)**exponent*(sigma_initial-sigma_final)+sigma_final
        fmin,fmax=fitness.min(),fitness.max(); seeds=[]
        for i in range(len(population)):
            ratio=1 if fmax==fmin else (fmax-fitness[i])/(fmax-fmin)
            n=int(round(min_seeds+ratio*(max_seeds-min_seeds)))
            for _ in range(n): seeds.append(np.clip(population[i]+rng.normal(0,sigma,n_dim),lo,hi))
        if seeds:
            seeds=np.asarray(seeds); sf=evaluate_population(seeds,items,capacity); combined=np.vstack((population,seeds)); cf=np.concatenate((fitness,sf))
        else: combined,cf=population,fitness
        order=np.argsort(cf)[:pop_max]; population,fitness=combined[order],cf[order]
        idx=np.argmin(fitness); best_history.append(fitness[idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    idx=np.argmin(fitness); run_iwo.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_iwo(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (IWO on Knapsack) ===")
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
    plt.title("IWO convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

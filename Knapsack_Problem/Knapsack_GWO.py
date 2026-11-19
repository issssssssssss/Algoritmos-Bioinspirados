import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_gwo(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]
    for it in range(n_iterations):
        order=np.argsort(fitness); alpha,beta,delta=population[order[:3]]
        a=2-2*it/max(1,n_iterations-1); new=np.empty_like(population)
        for i in range(pop_size):
            r1,r2=rng.rand(n_dim),rng.rand(n_dim); A1=2*a*r1-a; C1=2*r2
            r1,r2=rng.rand(n_dim),rng.rand(n_dim); A2=2*a*r1-a; C2=2*r2
            r1,r2=rng.rand(n_dim),rng.rand(n_dim); A3=2*a*r1-a; C3=2*r2
            X1=alpha-A1*np.abs(C1*alpha-population[i]); X2=beta-A2*np.abs(C2*beta-population[i]); X3=delta-A3*np.abs(C3*delta-population[i])
            new[i]=np.clip((X1+X2+X3)/3,lo,hi)
        population=new; fitness=evaluate_population(population,items,capacity)
        idx=np.argmin(fitness); best_history.append(fitness[idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    idx=np.argmin(fitness); run_gwo.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_gwo(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (GWO on Knapsack) ===")
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
    plt.title("GWO convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

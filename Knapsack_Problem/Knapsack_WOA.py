import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_woa(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),b=1.0,seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]
    best_idx=np.argmin(fitness); best_x=population[best_idx].copy(); best_f=fitness[best_idx]
    for it in range(n_iterations):
        best_idx=np.argmin(fitness); Xstar=population[best_idx].copy(); a=2-2*it/max(1,n_iterations-1); new=np.empty_like(population)
        for i in range(pop_size):
            X=population[i]; r1,r2=rng.rand(n_dim),rng.rand(n_dim); A=2*a*r1-a; C=2*r2; p=rng.rand()
            if p<.5:
                if np.linalg.norm(A)<1:
                    D=np.abs(C*Xstar-X); Xnew=Xstar-A*D
                else:
                    Xrand=population[rng.randint(pop_size)]; D=np.abs(C*Xrand-X); Xnew=Xrand-A*D
            else:
                l=rng.uniform(-1,1,n_dim); D=np.abs(Xstar-X); Xnew=D*np.exp(b*l)*np.cos(2*np.pi*l)+Xstar
            new[i]=np.clip(Xnew,lo,hi)
        population=new; fitness=evaluate_population(population,items,capacity)
        idx=np.argmin(fitness)
        if fitness[idx]<best_f: best_x,best_f=population[idx].copy(),fitness[idx]
        best_history.append(best_f); mean_history.append(np.mean(fitness)); best_position_history.append(best_x.copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-best_f:.6f} | Mean objective: {mean_history[-1]:.6f}")
    run_woa.evaluation_history=common.evaluation_history
    return result_position(best_x),best_f,best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_woa(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (WOA on Knapsack) ===")
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
    plt.title("WOA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

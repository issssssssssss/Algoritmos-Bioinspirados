import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_pfa(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),selection_rate=.3,max_seeds=5,sigma_initial=.5,sigma_final=.001,exponent=2,pollination_rate=.1,
    seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]
    for it in range(n_iterations):
        sigma=((n_iterations-it)/n_iterations)**exponent*(sigma_initial-sigma_final)+sigma_final
        order=np.argsort(fitness); selected=order[:max(1,int(selection_rate*pop_size))]
        fmin,fmax=fitness[selected].min(),fitness[selected].max(); seeds=[]
        for idx in selected:
            ratio=1 if fmax==fmin else (fmax-fitness[idx])/(fmax-fmin)
            for _ in range(max(1,int(round(ratio*max_seeds)))): seeds.append(np.clip(population[idx]+rng.normal(0,sigma,n_dim),lo,hi))
        seeds=np.asarray(seeds) if seeds else np.empty((0,n_dim))
        sf=evaluate_population(seeds,items,capacity) if len(seeds) else np.empty(0)
        for idx in order[-max(1,int(pollination_rate*pop_size)):]:
            donor=population[selected[rng.randint(len(selected))]]
            population[idx]=np.clip(.5*donor+.5*rng.uniform(lo,hi,n_dim),lo,hi)
            fitness[idx]=evaluate(population[idx],items,capacity)
        combined=np.vstack((population,seeds)) if len(seeds) else population
        cf=np.concatenate((fitness,sf)) if len(seeds) else fitness
        keep=np.argsort(cf)[:pop_size]; population,fitness=combined[keep],cf[keep]
        idx=np.argmin(fitness); best_history.append(fitness[idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    idx=np.argmin(fitness); run_pfa.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_pfa(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (PFA on Knapsack) ===")
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
    plt.title("PFA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

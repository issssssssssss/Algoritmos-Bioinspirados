import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_iwd(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),n_candidates=5,av=1.0,bv=.01,cv=1.0,init_vel=1.0,radius_initial=1.5,radius_final=.01,pull_strength=.05,
    seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    velocity=np.full(pop_size,init_vel); best_history=[]; mean_history=[]; best_position_history=[]
    for it in range(n_iterations):
        radius=radius_initial+(radius_final-radius_initial)*it/max(1,n_iterations-1)
        new=population.copy()
        for i in range(pop_size):
            candidates=[]
            for _ in range(n_candidates):
                direction=rng.uniform(-1,1,n_dim)
                direction/=np.linalg.norm(direction)+1e-12
                candidate=np.clip(population[i]+radius*direction*velocity[i],lo,hi)
                cf=evaluate(candidate,items,capacity)
                candidates.append((cf,candidate))
            cf,candidate=min(candidates,key=lambda z:z[0])
            if cf<fitness[i]:
                new[i]=candidate; velocity[i]=min(velocity[i]*1.1,init_vel*10)
            else:
                velocity[i]=max(velocity[i]*.9,1e-6)
        population=new; fitness=evaluate_population(population,items,capacity)
        idx=np.argmin(fitness); best_history.append(fitness[idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
    idx=np.argmin(fitness); run_iwd.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_iwd(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (IWD on Knapsack) ===")
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
    plt.title("IWD convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

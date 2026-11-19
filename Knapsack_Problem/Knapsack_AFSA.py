import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_afsa(
    n_dim=10, pop_size=30, n_iterations=300, bounds=(0, 1),
    visual=0.25, step=0.1, try_number=5, crowding_factor=0.6,
    seed=None, verbose=True, items=None, capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity)
    best_history=[]; mean_history=[]; best_position_history=[]
    best_idx=np.argmin(fitness); best_x=population[best_idx].copy(); best_f=fitness[best_idx]
    for it in range(n_iterations):
        progress=it/max(1,n_iterations-1); visual_t=visual*(1-.9*progress); step_t=step*(1-.9*progress)
        new=population.copy()
        for i in range(pop_size):
            Xi=population[i]; d=np.linalg.norm(population-Xi,axis=1); neigh=np.where((d>0)&(d<=visual_t))[0]
            candidates=[]
            if len(neigh)>0 and len(neigh)/pop_size<crowding_factor:
                center=population[neigh].mean(axis=0)
                if evaluate(center,items,capacity)<fitness[i]:
                    direction=center-Xi; norm=np.linalg.norm(direction)
                    if norm>1e-12:
                        m=np.clip(Xi+step_t*rng.rand()*direction/norm,lo,hi)
                        candidates.append((m,evaluate(m,items,capacity)))
                bn=neigh[np.argmin(fitness[neigh])]
                if fitness[bn]<fitness[i]:
                    direction=population[bn]-Xi; norm=np.linalg.norm(direction)
                    if norm>1e-12:
                        m=np.clip(Xi+step_t*rng.rand()*direction/norm,lo,hi)
                        candidates.append((m,evaluate(m,items,capacity)))
            if candidates: new[i]=min(candidates,key=lambda z:z[1])[0]; continue
            improved=False
            for _ in range(try_number):
                xj=np.clip(Xi+visual_t*rng.uniform(-1,1,n_dim),lo,hi)
                fj=evaluate(xj,items,capacity)
                if fj<fitness[i]:
                    direction=xj-Xi; norm=np.linalg.norm(direction)
                    new[i]=np.clip(Xi+step_t*rng.rand()*direction/(norm+1e-12),lo,hi); improved=True; break
            if not improved: new[i]=np.clip(Xi+step_t*rng.uniform(-1,1,n_dim),lo,hi)
        population=new; fitness=evaluate_population(population,items,capacity)
        idx=np.argmin(fitness)
        if fitness[idx]<best_f: best_x,best_f=population[idx].copy(),fitness[idx]
        best_history.append(best_f); mean_history.append(np.mean(fitness)); best_position_history.append(best_x.copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-best_f:.6f} | Mean objective: {mean_history[-1]:.6f}")
    run_afsa.evaluation_history=common.evaluation_history
    return result_position(best_x),best_f,best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_afsa(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (AFSA on Knapsack) ===")
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
    plt.title("AFSA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

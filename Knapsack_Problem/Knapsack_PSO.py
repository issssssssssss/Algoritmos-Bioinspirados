import numpy as np
import knapsack_common as common


def _continuous_to_binary(x, threshold=0.5):
    return (np.asarray(x) >= threshold).astype(int)


def run_pso(
    n_dim=10,n_particles=50,n_iterations=300,bounds=(0,1),w=.7,c1=1.5,c2=1.5,v_max=None,seed=None,verbose=True,items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds
    positions=rng.uniform(lo,hi,(n_particles,n_dim)); vmax=.2*(hi-lo) if v_max is None else v_max
    velocities=rng.uniform(-vmax,vmax,(n_particles,n_dim))
    pbest=positions.copy(); pbest_f=common.evaluate_population(positions,items,capacity,decode_fn=_continuous_to_binary)
    idx=np.argmin(pbest_f); gbest=pbest[idx].copy(); gbest_f=pbest_f[idx]
    best_history=[]; best_position_history=[]
    for it in range(n_iterations):
        r1,r2=rng.rand(n_particles,n_dim),rng.rand(n_particles,n_dim)
        velocities=np.clip(w*velocities+c1*r1*(pbest-positions)+c2*r2*(gbest-positions),-vmax,vmax)
        positions=np.clip(positions+velocities,lo,hi)
        f=common.evaluate_population(positions,items,capacity,decode_fn=_continuous_to_binary)
        improved=f<pbest_f; pbest[improved]=positions[improved]; pbest_f[improved]=f[improved]
        idx=np.argmin(pbest_f)
        if pbest_f[idx]<gbest_f: gbest,gbest_f=pbest[idx].copy(),pbest_f[idx]
        best_history.append(gbest_f); best_position_history.append(gbest.copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-gbest_f:.6f}")
    run_pso.evaluation_history=common.evaluation_history
    return common.result_position(gbest),gbest_f,best_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, best_pos_hist = run_pso(
        n_dim=len(DEFAULT_ITEMS),
        n_particles=50,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (PSO on Knapsack) ===")
    print("Best selection:", best_x)
    total_weight = sum(DEFAULT_ITEMS[i][0] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    total_value = sum(DEFAULT_ITEMS[i][1] for i in range(len(DEFAULT_ITEMS)) if int(best_x[i]) == 1)
    print("Total weight:", total_weight)
    print("Total value:", total_value)
    print("Best value:", -best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("PSO convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
from knapsack_common import evaluate, evaluate_population, decode_position, result_position
import knapsack_common as common


def run_bfoa(
    n_dim=10,pop_size=30,n_iterations=300,bounds=(0,1),
    Nc=10,Ns=4,Nre=4,Ped=.25,C=.1,seed=None,verbose=True,
    items=None,capacity=None,
):
    common.reset(); rng=np.random.RandomState(seed); lo,hi=bounds; step_size=C*(hi-lo)
    population=rng.uniform(lo,hi,(pop_size,n_dim)); fitness=evaluate_population(population,items,capacity); health=np.zeros(pop_size)
    best_history=[]; mean_history=[]; best_position_history=[]; reproduction_count=0
    def jcc(x,pop):
        d=pop-x; sq=np.sum(d*d,axis=1)
        return np.sum(-.1*np.exp(-.2*sq)+.1*np.exp(-10*sq))
    for it in range(n_iterations):
        for i in range(pop_size):
            delta=rng.uniform(-1,1,n_dim); delta/=np.linalg.norm(delta)+1e-12
            current=fitness[i]+jcc(population[i],population)
            new=np.clip(population[i]+step_size*delta,lo,hi); nf=evaluate(new,items,capacity)+jcc(new,population)
            if nf<current:
                population[i]=new; fitness[i]=evaluate(new,items,capacity)
                for _ in range(Ns):
                    trial=np.clip(population[i]+step_size*delta,lo,hi); tf=evaluate(trial,items,capacity)+jcc(trial,population)
                    if tf<fitness[i]+jcc(population[i],population): population[i]=trial; fitness[i]=evaluate(trial,items,capacity)
                    else: break
            health[i]+=fitness[i]
        idx=np.argmin(fitness); best_history.append(fitness[idx]); mean_history.append(np.mean(fitness)); best_position_history.append(population[idx].copy()); common.evaluation_history.append(common.evaluation_count)
        if verbose and (it%30==0 or it==n_iterations-1): print(f"Iter {it:4d} | Best value: {-fitness[idx]:.6f} | Mean objective: {mean_history[-1]:.6f}")
        if (it+1)%Nc==0:
            order=np.argsort(health); keep=order[:pop_size//2]; weak=order[pop_size//2:]
            population[weak]=population[keep]; fitness[weak]=fitness[keep]; health[:]=0; reproduction_count+=1
            if reproduction_count%Nre==0:
                for i in range(pop_size):
                    if rng.rand()<Ped:
                        population[i]=rng.uniform(lo,hi,n_dim); fitness[i]=evaluate(population[i],items,capacity)
    idx=np.argmin(fitness); run_bfoa.evaluation_history=common.evaluation_history
    return result_position(population[idx]),fitness[idx],best_history,mean_history,best_position_history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS

    best_x, best_f, best_hist, mean_hist, best_pos_hist = run_bfoa(
        n_dim=len(DEFAULT_ITEMS),
        pop_size=30,
        n_iterations=200,
        items=DEFAULT_ITEMS,
        capacity=DEFAULT_CAPACITY,
        seed=42,
        verbose=False,
    )

    print("\n=== Result (BFOA on Knapsack) ===")
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
    plt.title("BFOA convergence on Knapsack")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

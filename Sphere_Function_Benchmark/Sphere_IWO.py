"""
Invasive Weed Optimization (IWO) applied to the Sphere benchmark
function (Mehrabian & Lucas, 2006).

IWO mimics how invasive weeds colonize farmland: successful plants
produce more seeds, seeds scatter randomly around the parent (more
tightly over time), and the whole population is periodically thinned
back down to a maximum size, keeping only the fittest.

FITNESS: same direct sphere(x) evaluation of each plant's position.

PHASES (every iteration):

1. REPRODUCTION (seed count proportional to fitness)
   Each plant produces a number of seeds LINEARLY scaled between
   `min_seeds` and `max_seeds` based on how good its fitness is relative
   to the current population's best and worst:
       ratio = (f_worst - f_i) / (f_worst - f_best)
       n_seeds_i = min_seeds + ratio * (max_seeds - min_seeds)
   The best plant in the population produces max_seeds; the worst
   produces only min_seeds. This is IWO's direct analogue to natural
   selection -- fitter plants get proportionally more offspring.

2. SPATIAL DISPERSAL (seeds scatter around the parent)
   Each seed is placed at the parent's position plus Gaussian noise:
       seed = parent + N(0, sigma_t)
   where sigma_t (the standard deviation / dispersal radius) SHRINKS
   over the course of the run via a nonlinear schedule:
       sigma_t = ((n_iter - it)/n_iter)^exponent * (sigma_initial - sigma_final) + sigma_final
   Early iterations scatter seeds far from the parent (broad
   exploration); late iterations scatter them very close (fine
   exploitation) -- exactly the same exploration->exploitation idea as
   GWO/WOA's `a`, but implemented as a shrinking Gaussian radius instead
   of a shrinking pull-strength.

3. COMPETITIVE EXCLUSION (population capping)
   All existing plants and all newly produced seeds are pooled together,
   sorted by fitness, and only the best `pop_max` are kept -- everyone
   else, parent or seed, is discarded. This is what keeps the total
   population size bounded despite unlimited seed production, and it's
   also IWO's only real "selection pressure" mechanism -- there's no
   separate mutation or crossover step, just "produce lots of variation,
   then keep only the best."
"""

import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return sphere(x)


def evaluate_population(population):
    return np.array([evaluate(ind) for ind in population])


def sphere(x):
    return np.sum(x ** 2)


def run_iwo(
    n_dim=10,
    pop_size=30,        # also used as pop_max (population cap)
    n_iterations=200,
    bounds=(-5.12, 5.12),
    min_seeds=0,
    max_seeds=5,
    exponent=2,          # nonlinear modulation index for sigma decay
    sigma_initial=0.5,
    sigma_final=0.001,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds
    pop_max = pop_size

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = evaluate_population(population)

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        sigma_t = ((n_iterations - it) / n_iterations) ** exponent * \
            (sigma_initial - sigma_final) + sigma_final

        f_min, f_max = fitness.min(), fitness.max()
        all_seeds = []
        for i in range(len(population)):
            if f_max == f_min:
                n_seeds = max_seeds
            else:
                ratio = (f_max - fitness[i]) / (f_max - f_min)
                n_seeds = int(round(min_seeds + ratio * (max_seeds - min_seeds)))
            for _ in range(n_seeds):
                s = population[i] + rng.normal(0, sigma_t, size=n_dim)
                all_seeds.append(np.clip(s, lo, hi))

        if all_seeds:
            all_seeds = np.array(all_seeds)
            seed_fitness = evaluate_population(all_seeds)
            combined = np.vstack([population, all_seeds])
            combined_fitness = np.concatenate([fitness, seed_fitness])
        else:
            combined, combined_fitness = population, fitness

        order = np.argsort(combined_fitness)[:pop_max]
        population = combined[order]
        fitness = combined_fitness[order]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_iwo.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_iwo(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (IWO on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("IWO Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
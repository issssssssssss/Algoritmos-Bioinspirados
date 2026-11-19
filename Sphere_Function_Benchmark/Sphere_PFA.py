"""
Paddy Field Algorithm (PFA) applied to the Sphere benchmark function
(based on the description in Premaratne, Balasubramaniam & Sugathadasa,
2009).

IMPORTANT CAVEAT: unlike GWO/WOA/FA/ABC/IWO (which have one universally
cited canonical formula set), PFA is MUCH less standardized in the
literature -- there is no single widely-agreed-upon update equation the
way there is for, say, GWO's alpha/beta/delta blend. Different papers
implement "paddy field" ideas differently. What follows is a faithful
implementation of the algorithm's commonly-described STRUCTURE (seed
selection, proportional seeding, dispersal, pollination), built the same
way you'd build it from reading the original description, but treat the
specific formulas as a reasonable interpretation rather than "the"
official equations (the way GWO's formulas are).

PFA models rice paddy cultivation: only the best plants are chosen to
sow seeds; seeds scatter near their parent; and a separate
"pollination" process occasionally moves poor plants toward a mix of a
good plant and a random new location, representing wind/insect-carried
cross-pollination introducing fresh genetic material from elsewhere in
the field.

FITNESS: same direct sphere(x) evaluation.

PHASES (every iteration):

1. SELECTION (seeding plants)
   Sort the population by fitness and select the top `selection_rate`
   fraction as "seeding plants" -- only these will produce seeds this
   round (unlike IWO, where EVERY plant produces at least a few seeds).

2. SEED PRODUCTION & DISPERSAL
   Each seeding plant produces a number of seeds proportional to its
   rank within the selected group (best selected plant produces the
   most seeds), scattered around it with Gaussian noise whose radius
   shrinks over the run -- same shrinking-dispersal idea as IWO.

3. POLLINATION (global exploration)
   A `pollination_rate` fraction of the WORST plants in the population
   are discarded and replaced by a blend of a randomly chosen seeding
   plant and a completely random new location:
       new_plant = 0.5 * (random selected plant) + 0.5 * (random point)
   This is PFA's mechanism for injecting long-range exploration --
   comparable in spirit to IWO's population-cap "start fresh" effect,
   or ABC's scout bees, but implemented as a directed half-jump instead
   of a fully random restart.

4. POPULATION CONTROL
   All original plants, new seeds, and pollinated replacements are
   pooled and the best `pop_size` are kept, capping the population back
   to its original size (same idea as IWO's competitive exclusion).
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


def run_pfa(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
    selection_rate=0.3,     # fraction of plants selected as seeding plants
    max_seeds=5,
    sigma_initial=0.5,
    sigma_final=0.001,
    exponent=2,
    pollination_rate=0.1,   # fraction of population replaced via pollination each iteration
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = evaluate_population(population)

    best_history = []
    mean_history = []
    best_position_history = []

    n_select = max(1, int(selection_rate * pop_size))
    n_pollinate = max(1, int(pollination_rate * pop_size))

    for it in range(n_iterations):
        sigma_t = ((n_iterations - it) / n_iterations) ** exponent * \
            (sigma_initial - sigma_final) + sigma_final

        order = np.argsort(fitness)
        selected = order[:n_select]

        # --- 2. Seed production & dispersal ---
        f_min, f_max = fitness[selected].min(), fitness[selected].max()
        seeds = []
        for rank, idx in enumerate(selected):
            if f_max == f_min:
                n_seeds = max_seeds
            else:
                ratio = (f_max - fitness[idx]) / (f_max - f_min)
                n_seeds = max(1, int(round(ratio * max_seeds)))
            for _ in range(n_seeds):
                s = population[idx] + rng.normal(0, sigma_t, size=n_dim)
                seeds.append(np.clip(s, lo, hi))
        seeds = np.array(seeds) if seeds else np.empty((0, n_dim))
        seed_fitness = evaluate_population(seeds) if len(seeds) else np.empty(0)

        # --- 3. Pollination: replace worst plants with a directed jump ---
        worst_idx = order[-n_pollinate:]
        for idx in worst_idx:
            donor = population[selected[rng.randint(len(selected))]]
            random_component = rng.uniform(lo, hi, size=n_dim)
            pollinated = 0.5 * donor + 0.5 * random_component
            population[idx] = np.clip(pollinated, lo, hi)
            fitness[idx] = evaluate(population[idx])

        # --- 4. Population control ---
        combined = np.vstack([population, seeds]) if len(seeds) else population
        combined_fitness = np.concatenate([fitness, seed_fitness]) if len(seeds) else fitness
        keep_order = np.argsort(combined_fitness)[:pop_size]
        population = combined[keep_order]
        fitness = combined_fitness[keep_order]

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_pfa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_pfa(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (PFA on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("PFA Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
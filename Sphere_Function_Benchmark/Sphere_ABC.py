"""
Artificial Bee Colony (ABC) applied to the Sphere benchmark function.

ABC (Karaboga, 2005) models a bee colony's foraging behavior. Each
"food source" is a candidate solution. There are three types of bees,
applied as three distinct PHASES every iteration -- this is the most
GA-unlike algorithm in the set, since it has explicit role division
rather than a single position-update rule:

1. EMPLOYED BEE PHASE
   Each employed bee is permanently assigned to one food source i. It
   searches the immediate NEIGHBORHOOD of that source by picking one
   random dimension j and one random other source k, then proposing:
       v_i[j] = x_i[j] + phi * (x_i[j] - x_k[j]),   phi ~ Uniform(-1, 1)
   Only dimension j changes -- this is a local, single-dimension probe,
   not a full jump. GREEDY SELECTION: if v_i is better than x_i, replace
   it and reset its "trial" counter to 0; otherwise keep x_i and
   increment its trial counter (tracking how long this source has gone
   without improving).

2. ONLOOKER BEE PHASE
   Onlooker bees don't have a fixed source -- they watch the employed
   bees' "waggle dances" and PROBABILISTICALLY choose which source to
   investigate, favoring better (fitter) sources:
       p_i = fitness_i / sum(fitness)
   Once a source is chosen via roulette-wheel selection, the onlooker
   does the exact same neighborhood search + greedy selection as phase 1.
   This is what channels more search effort toward promising regions.

3. SCOUT BEE PHASE
   Any food source whose trial counter exceeds a `limit` (i.e. it hasn't
   improved in `limit` consecutive attempts) is considered EXHAUSTED and
   abandoned. Its employed bee becomes a scout and the source is
   replaced with a fresh random point in the search space, trial reset
   to 0. This is ABC's mechanism for escaping local optima -- similar in
   spirit to a GA's mutation, but triggered by stagnation rather than by
   chance every generation.

The colony's best-ever solution is tracked separately from the food
sources themselves, since a scout replacing a source can (temporarily)
lose good solutions from the working population.
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


def run_abc(
    n_dim=10,
    pop_size=30,        # number of food sources (= number of employed bees = number of onlooker bees)
    n_iterations=200,
    bounds=(-5.12, 5.12),
    limit=50,            # trials before a source is abandoned
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
    trial = np.zeros(pop_size, dtype=int)

    def fitness_transform(f):
        # ABC's selection probabilities are defined for fitness where
        # HIGHER = better, but sphere is a minimization problem, so we
        # transform: higher transformed value = lower (better) raw cost.
        return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))

    def neighbor_search(i):
        j = rng.randint(n_dim)
        k = rng.choice([idx for idx in range(pop_size) if idx != i])
        phi = rng.uniform(-1, 1)
        v = population[i].copy()
        v[j] = population[i][j] + phi * (population[i][j] - population[k][j])
        v = np.clip(v, lo, hi)
        return v

    best_history = []
    mean_history = []
    best_position_history = []
    best_ever_x = population[np.argmin(fitness)].copy()
    best_ever_f = fitness.min()

    for it in range(n_iterations):
        # --- 1. Employed bee phase ---
        for i in range(pop_size):
            v = neighbor_search(i)
            fv = evaluate(v)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        # --- 2. Onlooker bee phase ---
        trans_fit = fitness_transform(fitness)
        probs = trans_fit / trans_fit.sum()
        for _ in range(pop_size):
            i = rng.choice(pop_size, p=probs)
            v = neighbor_search(i)
            fv = evaluate(v)
            if fv < fitness[i]:
                population[i], fitness[i] = v, fv
                trial[i] = 0
            else:
                trial[i] += 1

        # --- 3. Scout bee phase ---
        for i in range(pop_size):
            if trial[i] > limit:
                population[i] = rng.uniform(lo, hi, size=n_dim)
                fitness[i] = evaluate(population[i])
                trial[i] = 0

        # --- Track best-ever ---
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_ever_f:
            best_ever_f = fitness[gen_best_idx]
            best_ever_x = population[gen_best_idx].copy()

        best_history.append(best_ever_f)
        mean_history.append(np.mean(fitness))
        best_position_history.append(best_ever_x.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best-ever: {best_ever_f:.8f} | Mean: {mean_history[-1]:.6f}")

    run_abc.evaluation_history = evaluation_history
    return best_ever_x, best_ever_f, best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_abc(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (ABC on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best-ever fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("ABC Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""
Artificial Fish Swarm Algorithm (AFSA) applied to the Sphere benchmark
function.

AFSA (Li, 2002) mimics how fish find food by combining individual
foraging with social behaviors (following the crowd, following the best
fish). Unlike GWO/WOA/FA (one fixed movement rule applied every time),
each fish tries MULTIPLE behaviors and only commits to a move if it
actually helps.

FITNESS: same direct evaluation, f(x) = sphere(x). "Food concentration"
in the original biological metaphor = better (lower) fitness here.

PHASES (evaluated in this priority order, for each fish, each iteration):

1. SWARMING BEHAVIOR
   Find all other fish within `visual` distance (the fish's local view
   radius). If there are any neighbors, compute their center position.
   If the center is BETTER than the fish's current position AND the
   neighborhood isn't overcrowded (fraction of population within visual
   range < crowding_factor), move a step toward that center. This is
   what makes fish cluster together in promising regions rather than
   scattering to the individually-best spot each found.

2. FOLLOWING BEHAVIOR
   Among the same visual-range neighbors, find whichever ONE fish has
   the best fitness. If that fish is better than the current one (and
   again not overcrowded), move a step toward it. This channels the
   fish toward locally-best-known food.
   The fish independently computes both #1 and #2 and takes whichever
   move actually produces the better resulting position.

3. PREYING (individual foraging) -- used if swarming/following didn't
   trigger (e.g. no neighbors, or too crowded)
   Randomly try up to `try_number` candidate points within visual range;
   if any candidate is better than the current position, move a step
   toward the FIRST such improving candidate found.

4. RANDOM MOVEMENT -- last resort
   If no improving neighbor or candidate was found by any of the above,
   move a small random step. This keeps fish exploring even when
   locally "stuck."

The `visual` radius (how far a fish can sense) and `step` (how far it
can move per iteration) are the two key parameters controlling
exploration vs exploitation, similar in spirit to gamma/beta0 in FA.

PRACTICAL NOTE: the textbook AFSA formulation uses a FIXED visual/step
size throughout the run. In practice this means it converges into the
right neighborhood quickly but then can't refine further -- it overshoots
past the optimum by roughly `step` every time, unable to take smaller
steps once close. A very standard practical fix (used here) is to let
`visual` and `step` shrink over the course of the run, exactly like
GWO/WOA's decaying `a` parameter -- broad exploration early on, precise
exploitation later.
"""

import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return sphere(x)


def sphere(x):
    return np.sum(x ** 2)


def run_afsa(
    n_dim=10,
    pop_size=30,
    n_iterations=200,
    bounds=(-5.12, 5.12),
    visual=2.0,
    step=0.5,
    try_number=5,
    crowding_factor=0.6,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)
    lo, hi = bounds

    population = rng.uniform(lo, hi, size=(pop_size, n_dim))
    fitness = np.array([evaluate(ind) for ind in population])

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        # shrink visual range and step size over time -- lets AFSA explore
        # broadly early on, then refine precisely later (same
        # exploration->exploitation idea as GWO/WOA's decaying `a`)
        progress = it / n_iterations
        visual_t = visual * (1 - 0.9 * progress)
        step_t = step * (1 - 0.9 * progress)

        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(pop_size):
            Xi = population[i]
            dists = np.linalg.norm(population - Xi, axis=1)
            neighbor_idx = np.where((dists > 0) & (dists <= visual_t))[0]

            candidate_moves = []  # (resulting_position, resulting_fitness)

            # --- 1. Swarming ---
            if len(neighbor_idx) > 0 and len(neighbor_idx) / pop_size < crowding_factor:
                center = population[neighbor_idx].mean(axis=0)
                center_fitness = evaluate(center)
                if center_fitness < fitness[i]:
                    direction = center - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, evaluate(move)))

            # --- 2. Following ---
            if len(neighbor_idx) > 0 and len(neighbor_idx) / pop_size < crowding_factor:
                best_neighbor = neighbor_idx[np.argmin(fitness[neighbor_idx])]
                if fitness[best_neighbor] < fitness[i]:
                    direction = population[best_neighbor] - Xi
                    norm = np.linalg.norm(direction)
                    if norm > 1e-12:
                        move = Xi + step_t * rng.rand() * direction / norm
                        move = np.clip(move, lo, hi)
                        candidate_moves.append((move, evaluate(move)))

            if candidate_moves:
                # take whichever behavior gave the better outcome
                best_move = min(candidate_moves, key=lambda m: m[1])
                new_population[i] = best_move[0]
                new_fitness[i] = best_move[1]
                continue

            # --- 3. Preying ---
            improved = False
            for _ in range(try_number):
                Xj = Xi + visual_t * rng.uniform(-1, 1, size=n_dim)
                Xj = np.clip(Xj, lo, hi)
                Xj_fitness = evaluate(Xj)
                if Xj_fitness < fitness[i]:
                    direction = Xj - Xi
                    norm = np.linalg.norm(direction)
                    move = Xi + step_t * rng.rand() * direction / (norm + 1e-12)
                    new_population[i] = np.clip(move, lo, hi)
                    new_fitness[i] = evaluate(new_population[i])
                    improved = True
                    break

            # --- 4. Random movement (last resort) ---
            if not improved:
                move = Xi + step_t * rng.uniform(-1, 1, size=n_dim)
                new_population[i] = np.clip(move, lo, hi)
                new_fitness[i] = evaluate(new_population[i])

        population = new_population
        fitness = new_fitness

        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])
        mean_history.append(np.mean(fitness))
        best_position_history.append(population[best_idx].copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {best_history[-1]:.8f} | Mean: {mean_history[-1]:.6f}")

    best_idx = np.argmin(fitness)
    run_afsa.evaluation_history = evaluation_history
    return population[best_idx], fitness[best_idx], best_history, mean_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, best_hist, mean_hist, pos_hist = run_afsa(
        n_dim=10, pop_size=30, n_iterations=200, bounds=(-5.12, 5.12), seed=42,
    )

    print("\n=== Result (AFSA on Sphere) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(best_hist, label="Best fitness")
    plt.plot(mean_hist, label="Mean fitness", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (log scale)")
    plt.title("AFSA Convergence on Sphere Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
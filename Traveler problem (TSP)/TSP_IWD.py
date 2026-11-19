"""
Intelligent Water Drops (IWD) applied to the TSP, using the NATIVE
graph-based formulation (Shah-Hosseini, 2007) -- NOT the continuous
adaptation used for sphere/Rastrigin.

This is the IWD equivalent of the ACO/ACOR distinction: TSP is one of
the problems IWD was ORIGINALLY tested on in Shah-Hosseini's paper
(alongside n-queens and robot path planning), so no encoding trick is
needed here -- cities are graph nodes, roads between them are edges,
and water genuinely can be thought of as flowing along them, eroding
soil as it passes. Compare this to sphere/Rastrigin, where we had to
invent a continuous "candidate generation" adaptation because no graph
exists in real-valued space.

PHASES (repeated every iteration):

1. SOIL INITIALIZATION (once, before the loop)
   A soil value soil[i][j] is maintained for every edge (city i -> city
   j), initialized to a uniform constant. Lower soil = easier to flow
   through = more attractive to future drops. This is IWD's analogue to
   ACO's pheromone matrix, but inverted: ants prefer HIGH pheromone,
   drops prefer LOW soil.

2. TOUR CONSTRUCTION (each of m drops builds one full tour)
   Each drop starts at a random city with an initial velocity, and
   repeatedly chooses its NEXT city from those not yet visited. The
   probability of choosing city j is INVERSELY related to the soil on
   edge (i, j) -- edges with less soil (well-traveled, previously good)
   are more likely to be chosen:
       P(i -> j) proportional to  1 / (soil[i][j] + epsilon)
   After choosing an edge, the drop's VELOCITY updates using IWD's
   canonical formula:
       v_new = v + av / (bv + cv * soil[i][j]^2)
   Lower soil on the edge taken => bigger velocity gain -- a drop that
   has been flowing through easy (low-soil) terrain moves faster.

3. LOCAL SOIL UPDATE (erosion, happens as each drop moves)
   The instant a drop uses edge (i, j), that edge's soil can optionally
   be reduced (eroded) by a further fraction, in addition to the global
   update below. In practice, testing on this instance showed that ANY
   local erosion on top of the global update made final results WORSE,
   not better -- local erosion applied by every drop's ordinary
   movement erodes edges faster than the differentiation between good
   and bad edges can build up, effectively adding noise that swamps the
   signal from the global update. The default here disables it
   (local_update_rate=0.0); it's left as a parameter so you can
   re-enable it and see this effect yourself.

4. GLOBAL SOIL UPDATE (reinforcement, after all drops finish)
   Once every drop has completed a full tour, the ITERATION-BEST drop's
   entire path gets an extra soil reduction, MULTIPLICATIVELY scaled by
   a fixed fraction:
       soil[i][j] *= (1 - global_update_rate)   for every edge (i,j) on the best tour
   This is the positive-feedback loop that drives convergence -- the
   best-found path becomes progressively easier to flow through, making
   it more attractive to future drops, very similar in spirit to ACO's
   pheromone deposit but working through REDUCTION instead of addition.
   (Using a multiplicative fraction rather than a flat subtraction of
   Q/length matters in practice -- a flat subtraction barely changes a
   large soil value and stalls convergence; scaling by soil's own
   current magnitude keeps the reinforcement meaningful regardless of
   scale.)
"""

import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def evaluate_tour(tour, dist_matrix):
    global evaluation_count
    evaluation_count += 1
    return tour_length(tour, dist_matrix)

def construct_tour(soil, n_cities, av, bv, cv, init_vel, local_update_rate, rng):
    start = rng.randint(n_cities)
    visited = [start]
    unvisited = set(range(n_cities)) - {start}
    velocity = init_vel

    while unvisited:
        current = visited[-1]
        candidates = list(unvisited)
        soils = np.array([soil[current, j] for j in candidates])
        desirability = 1.0 / (soils + 1e-6)
        probs = desirability / desirability.sum()
        next_city = rng.choice(candidates, p=probs)

        chosen_soil = soil[current, next_city]
        velocity = velocity + av / (bv + cv * chosen_soil ** 2)

        # local soil update (erosion) on the edge just used
        soil[current, next_city] *= (1 - local_update_rate)
        soil[next_city, current] *= (1 - local_update_rate)

        visited.append(next_city)
        unvisited.remove(next_city)

    return np.array(visited)


def run_iwd(
    dist_matrix,
    n_cities,
    n_drops=30,
    n_iterations=800,
    av=1.0, bv=0.01, cv=1.0,
    init_vel=200.0,
    init_soil=1000.0,
    local_update_rate=0.0,   # see note below -- any local erosion tested made results WORSE here
    global_update_rate=0.05,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)

    soil = np.full((n_cities, n_cities), float(init_soil))
    np.fill_diagonal(soil, np.inf)  # never "choose" staying at the same city

    best_history = []
    mean_history = []
    best_position_history = []
    best_ever_tour = None
    best_ever_len = np.inf

    for it in range(n_iterations):
        tours = [construct_tour(soil, n_cities, av, bv, cv, init_vel, local_update_rate, rng)
                  for _ in range(n_drops)]
        lengths = np.array([evaluate_tour(t, dist_matrix) for t in tours])

        gen_best_idx = np.argmin(lengths)
        if lengths[gen_best_idx] < best_ever_len:
            best_ever_len = lengths[gen_best_idx]
            best_ever_tour = tours[gen_best_idx].copy()

        # --- 4. Global soil update: reinforce the ITERATION-best drop's path ---
        # Multiplicative reduction (a fixed fraction of current soil) scales
        # correctly regardless of the soil magnitude -- an absolute
        # subtraction (soil -= Q/length) barely dents a large soil value and
        # was the original bug that kept this version from converging.
        best_tour_this_iter = tours[gen_best_idx]
        for k in range(n_cities):
            i, j = best_tour_this_iter[k], best_tour_this_iter[(k + 1) % n_cities]
            soil[i, j] *= (1 - global_update_rate)
            soil[j, i] = soil[i, j]

        best_history.append(best_ever_len)
        mean_history.append(np.mean(lengths))
        best_position_history.append(best_ever_tour.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best-ever tour length: {best_ever_len:.3f}")

    run_iwd.evaluation_history = evaluation_history
    return best_ever_tour, best_ever_len, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_iwd(
        dmat, N_CITIES, n_drops=30, n_iterations=800, seed=42,
    )

    print("\n=== Result (IWD on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best-ever")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("IWD (native, graph-based) Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()
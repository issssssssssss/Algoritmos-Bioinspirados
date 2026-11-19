"""
Ant Colony Optimization applied to the TSP, using the CLASSIC pheromone-
trail formulation (Ant System, Dorigo et al., 1996) -- NOT the
continuous ACOR adaptation used for sphere/Rastrigin.

This is worth pausing on: TSP is the problem ACO was originally
invented for. Cities are nodes, roads between them are edges, and real
ants really do find short paths by depositing pheromone as they walk.
Unlike sphere/Rastrigin (where we had to invent the random-key trick to
force a continuous algorithm onto a combinatorial-feeling encoding),
here the pheromone-on-edges idea maps onto the problem exactly as
originally intended -- no encoding trick needed.

PHASES (repeated every iteration):

1. PHEROMONE INITIALIZATION (once, before the loop)
   A pheromone value tau[i][j] is maintained for every directed edge
   (city i -> city j), initialized to a small uniform constant. This
   matrix IS the colony's shared memory -- unlike GA/PSO/etc., there is
   no population of solutions carried between iterations, only this
   edge-weight matrix.

2. TOUR CONSTRUCTION (each of m ants builds one full tour)
   Each ant starts at a random city and repeatedly chooses its NEXT city
   from the ones not yet visited, using a probability that combines
   pheromone strength and a greedy heuristic:
       P(i -> j) proportional to  tau[i][j]^alpha * eta[i][j]^beta
   where eta[i][j] = 1 / distance[i][j] is the "visibility" heuristic
   (favors nearby cities), alpha controls how much pheromone influences
   the choice, and beta controls how much raw distance influences it.
   This continues until all cities have been visited (a complete tour).

3. PHEROMONE EVAPORATION
   After all ants finish their tours, every edge's pheromone decays:
       tau[i][j] = (1 - rho) * tau[i][j]
   This is what prevents the colony from prematurely locking onto the
   first decent tour found -- old information fades unless reinforced.

4. PHEROMONE DEPOSIT (reinforcement)
   Each ant then deposits pheromone on the edges of the tour IT walked,
   in proportion to how good that tour was:
       tau[i][j] += Q / tour_length   for every edge (i,j) the ant used
   Shorter tours deposit MORE pheromone (since Q/length is larger),
   so better tours' edges get reinforced more strongly, making them more
   attractive to future ants -- this is the core positive-feedback loop
   that drives convergence.

Over many iterations, pheromone concentrates on the edges of good
tours, and the probabilistic city-choice rule increasingly favors them,
while evaporation keeps some exploration alive throughout the run.
"""

import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def evaluate_tour(tour, dist_matrix):
    global evaluation_count
    evaluation_count += 1
    return tour_length(tour, dist_matrix)

def construct_tour(pheromone, eta, n_cities, alpha, beta, rng):
    start = rng.randint(n_cities)
    visited = [start]
    unvisited = set(range(n_cities)) - {start}

    while unvisited:
        current = visited[-1]
        candidates = list(unvisited)
        weights = np.array([
            (pheromone[current, j] ** alpha) * (eta[current, j] ** beta)
            for j in candidates
        ])
        weights_sum = weights.sum()
        if weights_sum == 0 or not np.isfinite(weights_sum):
            probs = np.full(len(candidates), 1.0 / len(candidates))
        else:
            probs = weights / weights_sum
        next_city = rng.choice(candidates, p=probs)
        visited.append(next_city)
        unvisited.remove(next_city)

    return np.array(visited)


def run_aco(
    dist_matrix,
    n_cities,
    n_ants=30,
    n_iterations=300,
    alpha=1.0,    # pheromone influence
    beta=3.0,     # heuristic (distance) influence
    rho=0.3,      # evaporation rate
    Q=100.0,      # pheromone deposit constant
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)

    eta = 1.0 / (dist_matrix + np.eye(n_cities))  # avoid div-by-zero on diagonal
    np.fill_diagonal(eta, 0)
    pheromone = np.full((n_cities, n_cities), 1.0)

    best_history = []
    mean_history = []
    best_position_history = []
    best_ever_tour = None
    best_ever_len = np.inf

    for it in range(n_iterations):
        tours = [construct_tour(pheromone, eta, n_cities, alpha, beta, rng) for _ in range(n_ants)]
        lengths = np.array([evaluate_tour(t, dist_matrix) for t in tours])

        gen_best_idx = np.argmin(lengths)
        if lengths[gen_best_idx] < best_ever_len:
            best_ever_len = lengths[gen_best_idx]
            best_ever_tour = tours[gen_best_idx].copy()

        # --- 3. Evaporation ---
        pheromone *= (1 - rho)

        # --- 4. Deposit ---
        for tour, length in zip(tours, lengths):
            deposit = Q / length
            for k in range(n_cities):
                i, j = tour[k], tour[(k + 1) % n_cities]
                pheromone[i, j] += deposit
                pheromone[j, i] += deposit  # symmetric TSP

        best_history.append(best_ever_len)
        mean_history.append(np.mean(lengths))
        best_position_history.append(best_ever_tour.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best-ever tour length: {best_ever_len:.3f}")

    run_aco.evaluation_history = evaluation_history
    return best_ever_tour, best_ever_len, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, best_hist, mean_hist, pos_hist = run_aco(
        dmat, N_CITIES, n_ants=30, n_iterations=300, seed=42,
    )

    print("\n=== Result (ACO on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(best_hist, label="Best-ever")
    axes[0].plot(mean_hist, label="Mean", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Tour length")
    axes[0].set_title("ACO (Ant System) Convergence on TSP")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()
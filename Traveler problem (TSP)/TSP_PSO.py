"""
Particle Swarm Optimization (PSO) applied to the TSP.
 
PSO natively works on continuous vectors, not permutations.
We use the same RANDOM-KEY idea (Bean, 1994): each particle's position
is a vector of n_cities continuous values ("keys"). The tour is decoded
by sorting city indices by their keys (argsort). PSO's velocity/position
updates happen entirely in this continuous key-space; only the fitness
evaluation needs the decoded tour.


Particle Swarm Optimization (PSO) — the general algorithm loop
Initialization — create a "swarm" of particles, each with a random position 
            (a candidate solution, as a vector of real numbers) and a random 
            small velocity vector.
Personal & global best tracking — evaluate every particle's fitness. 
            Each particle remembers its own best position ever found (pbest); 
            the whole swarm remembers the single best position anyone has found (gbest).
Velocity update — for each particle, update its velocity by blending three pulls: 
            keep some of its old velocity (inertia — momentum), add a pull toward its 
            own pbest (cognitive component — "trust your own experience"), and add a 
            pull toward the swarm's gbest (social component — "trust the group").
Position update — simply add the (just-updated) velocity to the particle's current 
            position: position = position + velocity. Since positions are continuous,
            this is a direct move through the search space — no transfer function or 
            coin-flip is needed here (that trick is specific to binary PSO, where a 
            sigmoid converts velocity into a probability of a bit being 1).
Re-evaluate — compute the fitness of each particle's new position.
Update bests — if a particle's new position beats its own pbest, update it; if it also
            beats the swarm's gbest, update that too.
Repeat steps 3–6 for a fixed number of iterations (or until gbest stops improving). 
            Unlike GA, there's no selection/crossover/replacement — every particle 
            survives every iteration and just keeps moving.
"""

import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def decode_keys(keys):
    return np.argsort(keys)


def fitness_of_keys(keys, dist_matrix):
    global evaluation_count
    tour = decode_keys(keys)
    evaluation_count += 1
    return tour_length(tour, dist_matrix)


def run_pso(
    dist_matrix,
    n_cities,
    n_particles=80,
    n_iterations=400,
    w=0.7,
    c1=1.5,
    c2=1.5,
    v_max=0.4,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    if seed is not None:
        np.random.seed(seed)

    lo, hi = 0.0, 1.0

    positions = np.random.uniform(lo, hi, size=(n_particles, n_cities))
    velocities = np.random.uniform(-v_max, v_max, size=(n_particles, n_cities))

    pbest_positions = positions.copy()
    pbest_fitness = np.array([fitness_of_keys(p, dist_matrix) for p in positions])

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    best_history = []

    for it in range(n_iterations):
        r1 = np.random.rand(n_particles, n_cities)
        r2 = np.random.rand(n_particles, n_cities)

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lo, hi)

        fitness = np.array([fitness_of_keys(p, dist_matrix) for p in positions])

        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]

        gen_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[gen_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[gen_best_idx]
            gbest_position = pbest_positions[gen_best_idx].copy()

        best_history.append(gbest_fitness)
        evaluation_history.append(evaluation_count)

        if verbose and (it % 40 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best tour length: {gbest_fitness:.3f}")

    best_tour = decode_keys(gbest_position)
    run_pso.evaluation_history = evaluation_history
    return best_tour, gbest_fitness, best_history



if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, history = run_pso(
        dmat, N_CITIES, n_particles=80, n_iterations=400, seed=42,
    )

    print("\n=== Result (PSO on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Global best tour length")
    axes[0].set_title("PSO Convergence on TSP")
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()
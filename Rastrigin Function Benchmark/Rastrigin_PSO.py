import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def rastrigin(x, A=10):
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return rastrigin(x)


def evaluate_population(population):
    return np.array([evaluate(ind) for ind in population])


def run_pso(
    n_dim=10,
    n_particles=50,
    n_iterations=300,
    bounds=(-5.12, 5.12),
    w=0.7,
    c1=1.5,
    c2=1.5,
    v_max=None,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    if seed is not None:
        np.random.seed(seed)

    lo, hi = bounds
    if v_max is None:
        v_max = 0.2 * (hi - lo)

    positions = np.random.uniform(lo, hi, size=(n_particles, n_dim))
    velocities = np.random.uniform(-v_max, v_max, size=(n_particles, n_dim))

    pbest_positions = positions.copy()
    pbest_fitness = evaluate_population(positions)

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    best_history = []
    best_position_history = []

    for it in range(n_iterations):
        r1 = np.random.rand(n_particles, n_dim)
        r2 = np.random.rand(n_particles, n_dim)

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        positions = positions + velocities
        positions = np.clip(positions, lo, hi)

        fitness = evaluate_population(positions)

        improved = fitness < pbest_fitness
        pbest_positions[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]

        gen_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[gen_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[gen_best_idx]
            gbest_position = pbest_positions[gen_best_idx].copy()

        best_history.append(gbest_fitness)
        best_position_history.append(gbest_position.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 30 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best: {gbest_fitness:.6f}")

    run_pso.evaluation_history = evaluation_history
    return gbest_position, gbest_fitness, best_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, history, position_history = run_pso(
        n_dim=10,
        n_particles=50,
        n_iterations=300,
        bounds=(-5.12, 5.12),
        seed=42,
    )

    print("\n=== Result (PSO on Rastrigin) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Global best fitness (log scale)")
    plt.title("PSO Convergence on Rastrigin Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

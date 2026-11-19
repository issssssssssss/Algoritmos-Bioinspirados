import numpy as np
import matplotlib.pyplot as plt

from Rastrigin_GA import run_ga
from Rastrigin_cGA import run_cga
from Rastrigin_PSO import run_pso


N_DIM = 10
BOUNDS = (-5.12, 5.12)
EVAL_BUDGET = 20000
SEED = 42


ga_pop = 100
ga_generations = EVAL_BUDGET // ga_pop

_, _, ga_best_hist, _, ga_best_pos_hist = run_ga(
    n_dim=N_DIM,
    pop_size=ga_pop,
    n_generations=ga_generations,
    bounds=BOUNDS,
    seed=SEED,
    verbose=False,
)

ga_best_pos_hist = np.array(ga_best_pos_hist)



cga_generations = EVAL_BUDGET // 2

_, _, cga_hist, cga_best_pos_hist = run_cga(
    n_dim=N_DIM,
    bits_per_var=16,
    pop_size=100,
    n_generations=cga_generations,
    bounds=BOUNDS,
    seed=SEED,
    verbose=False,
)

cga_best_pos_hist = np.array(cga_best_pos_hist)



pso_particles = 50
pso_iterations = EVAL_BUDGET // pso_particles

_, _, pso_hist, pso_best_pos_hist = run_pso(
    n_dim=N_DIM,
    n_particles=pso_particles,
    n_iterations=pso_iterations,
    bounds=BOUNDS,
    seed=SEED,
    verbose=False,
)

pso_best_pos_hist = np.array(pso_best_pos_hist)




x = np.linspace(BOUNDS[0], BOUNDS[1], 500)
y = np.linspace(BOUNDS[0], BOUNDS[1], 500)

X, Y = np.meshgrid(x, y)

A = 10

Z = (
    2 * A
    + X**2
    + Y**2
    - A * np.cos(2 * np.pi * X)
    - A * np.cos(2 * np.pi * Y)
)




plt.figure(figsize=(10, 8))

# Fondo de Rastrigin
contours = plt.contourf(
    X,
    Y,
    Z,
    levels=60,
    alpha=0.75
)

plt.colorbar(
    contours,
    label="Fitness"
)




plt.plot(
    ga_best_pos_hist[:, 0],
    ga_best_pos_hist[:, 1],
    "-o",
    markersize=2.5,
    linewidth=1,
    alpha=0.8,
    label="GA"
)




plt.plot(
    cga_best_pos_hist[:, 0],
    cga_best_pos_hist[:, 1],
    "-o",
    markersize=2.5,
    linewidth=1,
    alpha=0.8,
    label="cGA"
)




plt.plot(
    pso_best_pos_hist[:, 0],
    pso_best_pos_hist[:, 1],
    "-o",
    markersize=2.5,
    linewidth=1,
    alpha=0.8,
    label="PSO"
)




plt.scatter(
    0,
    0,
    s=250,
    marker="*",
    color="red",
    edgecolors="black",
    linewidths=1.5,
    zorder=10,
    label="Óptimo global (0,0)"
)




plt.scatter(
    ga_best_pos_hist[0, 0],
    ga_best_pos_hist[0, 1],
    s=40,
    marker="o",
    color="blue",
    edgecolors="black",
    zorder=9
)

plt.scatter(
    cga_best_pos_hist[0, 0],
    cga_best_pos_hist[0, 1],
    s=40,
    marker="o",
    color="orange",
    edgecolors="black",
    zorder=9
)

plt.scatter(
    pso_best_pos_hist[0, 0],
    pso_best_pos_hist[0, 1],
    s=40,
    marker="o",
    color="green",
    edgecolors="black",
    zorder=9
)



plt.scatter(
    ga_best_pos_hist[-1, 0],
    ga_best_pos_hist[-1, 1],
    s=40,
    marker="D",
    color="blue",
    edgecolors="black",
    zorder=10
)

plt.scatter(
    cga_best_pos_hist[-1, 0],
    cga_best_pos_hist[-1, 1],
    s=40,
    marker="D",
    color="orange",
    edgecolors="black",
    zorder=10
)

plt.scatter(
    pso_best_pos_hist[-1, 0],
    pso_best_pos_hist[-1, 1],
    s=40,
    marker="D",
    color="green",
    edgecolors="black",
    zorder=10
)




plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.title(
    f"Trayectoria de optimización sobre Rastrigin "
    f"(n_dim={N_DIM})"
)

plt.xlim(BOUNDS)
plt.ylim(BOUNDS)

plt.axhline(
    0,
    color="black",
    linewidth=0.5,
    alpha=0.4
)

plt.axvline(
    0,
    color="black",
    linewidth=0.5,
    alpha=0.4
)

plt.legend()

plt.grid(
    True,
    alpha=0.2
)

plt.tight_layout()
plt.show()
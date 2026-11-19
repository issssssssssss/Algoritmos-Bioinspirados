import os

import matplotlib.pyplot as plt
import numpy as np

from knapsack_common import DEFAULT_CAPACITY, DEFAULT_ITEMS
from Knapsack_ABC import run_abc
from Knapsack_ACOR import run_acor
from Knapsack_AFSA import run_afsa
from Knapsack_AIS import run_ais
from Knapsack_BFOA import run_bfoa
from Knapsack_cGA import run_cga
from Knapsack_FA import run_fa
from Knapsack_GA import run_ga
from Knapsack_GWO import run_gwo
from Knapsack_IWD import run_iwd
from Knapsack_IWO import run_iwo
from Knapsack_PFA import run_pfa
from Knapsack_PSO import run_pso
from Knapsack_WOA import run_woa

N_DIM = len(DEFAULT_ITEMS)
SEED = 42

EVAL_BUDGET_CLASSIC = 20000
EVAL_BUDGET_OTHERS = 6000

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def as_value_hist(history):
    hist = np.asarray(history, dtype=float)
    return -hist


# ---- GA / cGA / PSO ----
GA_POP = 100
ga_generations = EVAL_BUDGET_CLASSIC // GA_POP
_, _, ga_hist, _, _ = run_ga(
    n_dim=N_DIM,
    pop_size=GA_POP,
    n_generations=ga_generations,
    items=DEFAULT_ITEMS,
    capacity=DEFAULT_CAPACITY,
    seed=SEED,
    verbose=False,
)

CGA_POP = 100
cga_generations = EVAL_BUDGET_CLASSIC // 2
_, _, cga_hist, _ = run_cga(
    n_dim=N_DIM,
    pop_size=CGA_POP,
    n_generations=cga_generations,
    items=DEFAULT_ITEMS,
    capacity=DEFAULT_CAPACITY,
    seed=SEED,
    verbose=False,
)

PSO_PARTICLES = 50
pso_iterations = EVAL_BUDGET_CLASSIC // PSO_PARTICLES
_, _, pso_hist, _ = run_pso(
    n_dim=N_DIM,
    n_particles=PSO_PARTICLES,
    n_iterations=pso_iterations,
    items=DEFAULT_ITEMS,
    capacity=DEFAULT_CAPACITY,
    seed=SEED,
    verbose=False,
)

results = {
    "GA": (run_ga.evaluation_history, as_value_hist(ga_hist)),
    "cGA": (run_cga.evaluation_history, as_value_hist(cga_hist)),
    "PSO": (run_pso.evaluation_history, as_value_hist(pso_hist)),
}

# ---- Swarm / evolutionary variants ----
pop = 30
n_iter = EVAL_BUDGET_OTHERS // pop

_, _, gwo_hist, _, _ = run_gwo(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["GWO"] = (run_gwo.evaluation_history, as_value_hist(gwo_hist))

_, _, woa_hist, _, _ = run_woa(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["WOA"] = (run_woa.evaluation_history, as_value_hist(woa_hist))

_, _, abc_hist, _, _ = run_abc(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["ABC"] = (run_abc.evaluation_history, as_value_hist(abc_hist))

_, _, acor_hist, _, _ = run_acor(
    n_dim=N_DIM, archive_size=pop, n_ants=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["ACOR"] = (run_acor.evaluation_history, as_value_hist(acor_hist))

_, _, fa_hist, _, _ = run_fa(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["FA"] = (run_fa.evaluation_history, as_value_hist(fa_hist))

_, _, afsa_hist, _, _ = run_afsa(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["AFSA"] = (run_afsa.evaluation_history, as_value_hist(afsa_hist))

_, _, ais_hist, _, _ = run_ais(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["AIS"] = (run_ais.evaluation_history, as_value_hist(ais_hist))

_, _, bfoa_hist, _, _ = run_bfoa(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["BFOA"] = (run_bfoa.evaluation_history, as_value_hist(bfoa_hist))

_, _, iwo_hist, _, _ = run_iwo(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["IWO"] = (run_iwo.evaluation_history, as_value_hist(iwo_hist))

_, _, pfa_hist, _, _ = run_pfa(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["PFA"] = (run_pfa.evaluation_history, as_value_hist(pfa_hist))

_, _, iwd_hist, _, _ = run_iwd(
    n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
    items=DEFAULT_ITEMS, capacity=DEFAULT_CAPACITY,
    seed=SEED, verbose=False,
)
results["IWD"] = (run_iwd.evaluation_history, as_value_hist(iwd_hist))

print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best value':<15}")
for name, (evals, hist) in results.items():
    print(f"{name:<10} {evals[-1]:<14} {hist[-1]:<15.6f}")

plt.figure(figsize=(12, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
for (name, (evals, hist)), color in zip(results.items(), colors):
    plt.plot(evals, hist, label=name, color=color)

plt.xlabel("Fitness evaluations")
plt.ylabel("Best total value")
plt.title(f"Comparison of 14 algorithms on the Knapsack Problem (capacity={DEFAULT_CAPACITY})")
plt.legend(ncol=2, fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "comparison_14_knapsack.png"), dpi=150)
plt.show()

"""

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
    n_dim=N_DIM, pop_size=ga_pop, n_generations=ga_generations,
    bounds=BOUNDS, seed=SEED, verbose=False,
)
ga_evals = run_ga.evaluation_history

cga_generations = EVAL_BUDGET // 2
_, _, cga_hist, cga_best_pos_hist = run_cga(
    n_dim=N_DIM, bits_per_var=16, pop_size=100,
    n_generations=cga_generations, bounds=BOUNDS, seed=SEED, verbose=False,
)
cga_evals = run_cga.evaluation_history

pso_particles = 50
pso_iterations = EVAL_BUDGET // pso_particles
_, _, pso_hist, pso_best_pos_hist = run_pso(
    n_dim=N_DIM, n_particles=pso_particles, n_iterations=pso_iterations,
    bounds=BOUNDS, seed=SEED, verbose=False,
)
pso_evals = run_pso.evaluation_history





"""
"""
Compare GWO, WOA, ABC, and ACOR on the Rastrigin benchmark function
using a roughly equal number of fitness evaluations for a fair
comparison.
"""
"""
import numpy as np
import matplotlib.pyplot as plt

from Rastrigin_GWO import run_gwo
from Rastrigin_WOA import run_woa
from Rastrigin_ABC import run_abc
from Rastrigin_ACOR import run_acor


N_DIM = 10
BOUNDS = (-5.12, 5.12)
EVAL_BUDGET = 9000
SEED = 42

pop = 30
n_iter = EVAL_BUDGET // pop

_, _, gwo_hist, _, _ = run_gwo(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
                                 bounds=BOUNDS, seed=SEED, verbose=False)
_, _, woa_hist, _, _ = run_woa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
                                 bounds=BOUNDS, seed=SEED, verbose=False)
_, _, abc_hist, _, _ = run_abc(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter,
                                 bounds=BOUNDS, seed=SEED, verbose=False)
_, _, acor_hist, _, _ = run_acor(n_dim=N_DIM, archive_size=pop, n_ants=pop,
                                   n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)

evals = np.arange(1, n_iter + 1) * pop

print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best fitness':<15}")
print(f"{'GA':<10} {ga_evals[-1]:<14} {ga_best_hist[-1]:<15.6f}")
print(f"{'cGA':<10} {cga_evals[-1]:<14} {cga_hist[-1]:<15.6f}")
print(f"{'PSO':<10} {pso_evals[-1]:<14} {pso_hist[-1]:<15.6f}")
print(f"{'GWO':<10} {evals[-1]:<14} {gwo_hist[-1]:<15.6f}")
print(f"{'WOA':<10} {evals[-1]:<14} {woa_hist[-1]:<15.6f}")
print(f"{'ABC':<10} {evals[-1]:<14} {abc_hist[-1]:<15.6f}")
print(f"{'ACOR':<10} {evals[-1]:<14} {acor_hist[-1]:<15.6f}")

plt.figure(figsize=(9, 6))
plt.plot(ga_evals, ga_best_hist, label="GA (real-valued)")
plt.plot(cga_evals, cga_hist, label="cGA (compact GA, binary)")
plt.plot(pso_evals, pso_hist, label="PSO")
plt.plot(evals, gwo_hist, label="GWO")
plt.plot(evals, woa_hist, label="WOA")
plt.plot(evals, abc_hist, label="ABC")
plt.plot(evals, acor_hist, label="ACOR")
plt.yscale("log")
plt.xlabel("Fitness evaluations")
plt.ylabel("Best fitness found (log scale)")
plt.title(f"GWO vs WOA vs ABC vs ACOR on Rastrigin Function (n_dim={N_DIM})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


"""




import numpy as np
import matplotlib.pyplot as plt

from Rastrigin_GA import run_ga
from Rastrigin_cGA import run_cga
from Rastrigin_PSO import run_pso
from Rastrigin_GWO import run_gwo
from Rastrigin_WOA import run_woa
from Rastrigin_ABC import run_abc
from Rastrigin_ACOR import run_acor
from Rastrigin_FA import run_fa
from Rastrigin_AFSA import run_afsa
from Rastrigin_AIS import run_ais
from Rastrigin_BFOA import run_bfoa
from Rastrigin_IWO import run_iwo
from Rastrigin_PFA import run_pfa
from Rastrigin_IWD import run_iwd


N_DIM = 10
BOUNDS = (-5.12, 5.12)
SEED = 42

# Same two-budget setup used throughout: GA/cGA/PSO get a bigger budget
# since they need more evaluations to reach comparable precision; the
# swarm/bio-inspired algorithms converge fast enough that a smaller
# budget already shows the full picture. Set both to the same value if
# you'd rather compare everyone on one identical budget.
EVAL_BUDGET_CLASSIC = 20000   # GA, cGA, PSO
EVAL_BUDGET_OTHERS = 9000     # GWO, WOA, ABC, ACOR, FA, AFSA, AIS, BFOA, IWO, PFA, IWD

results = {}  # name -> (evals_array, best_history)

# ---- GA ----
ga_pop = 100
ga_generations = EVAL_BUDGET_CLASSIC // ga_pop
_, _, ga_best_hist, _, ga_best_pos_hist = run_ga(
    n_dim=N_DIM, pop_size=ga_pop, n_generations=ga_generations,
    bounds=BOUNDS, seed=SEED, verbose=False,
)
results["GA"] = (run_ga.evaluation_history, ga_best_hist)

# ---- cGA ---- (this algorithm's run_cga returns 4 values, no mean_history)
cga_generations = EVAL_BUDGET_CLASSIC // 2
_, _, cga_hist, cga_best_pos_hist = run_cga(
    n_dim=N_DIM, bits_per_var=16, pop_size=100,
    n_generations=cga_generations, bounds=BOUNDS, seed=SEED, verbose=False,
)
results["cGA"] = (run_cga.evaluation_history, cga_hist)

# ---- PSO ---- (this algorithm's run_pso also returns 4 values, no mean_history)
pso_particles = 50
pso_iterations = EVAL_BUDGET_CLASSIC // pso_particles
_, _, pso_hist, pso_best_pos_hist = run_pso(
    n_dim=N_DIM, n_particles=pso_particles, n_iterations=pso_iterations,
    bounds=BOUNDS, seed=SEED, verbose=False,
)
results["PSO"] = (run_pso.evaluation_history, pso_hist)

# ---- GWO / WOA / ABC / ACOR / FA / AFSA / AIS / BFOA / IWO / PFA / IWD ----
# All 11 of these return the full 5-tuple:
#   best_solution, best_fitness, best_history, mean_history, best_position_history
pop = 30
n_iter = EVAL_BUDGET_OTHERS // pop

_, _, gwo_hist, _, _ = run_gwo(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["GWO"] = (run_gwo.evaluation_history, gwo_hist)

_, _, woa_hist, _, _ = run_woa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["WOA"] = (run_woa.evaluation_history, woa_hist)

_, _, abc_hist, _, _ = run_abc(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["ABC"] = (run_abc.evaluation_history, abc_hist)

_, _, acor_hist, _, _ = run_acor(n_dim=N_DIM, archive_size=pop, n_ants=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["ACOR"] = (run_acor.evaluation_history, acor_hist)

_, _, fa_hist, _, _ = run_fa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["FA"] = (run_fa.evaluation_history, fa_hist)

_, _, afsa_hist, _, _ = run_afsa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["AFSA"] = (run_afsa.evaluation_history, afsa_hist)

_, _, ais_hist, _, _ = run_ais(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["AIS"] = (run_ais.evaluation_history, ais_hist)

_, _, bfoa_hist, _, _ = run_bfoa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["BFOA"] = (run_bfoa.evaluation_history, bfoa_hist)

_, _, iwo_hist, _, _ = run_iwo(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["IWO"] = (run_iwo.evaluation_history, iwo_hist)

_, _, pfa_hist, _, _ = run_pfa(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["PFA"] = (run_pfa.evaluation_history, pfa_hist)

_, _, iwd_hist, _, _ = run_iwd(n_dim=N_DIM, pop_size=pop, n_iterations=n_iter, bounds=BOUNDS, seed=SEED, verbose=False)
results["IWD"] = (run_iwd.evaluation_history, iwd_hist)

# ---- Print summary ----
print(f"{'Algorithm':<10} {'Evaluations':<14} {'Best fitness':<15}")
for name, (evals, hist) in results.items():
    print(f"{name:<10} {evals[-1]:<14} {hist[-1]:<15.6f}")

# ---- Plot ----
plt.figure(figsize=(11, 7))
colors = plt.cm.tab20(np.linspace(0, 1, len(results)))  # 14 distinct colors -- default cycle only has 10 and would repeat
for (name, (evals, hist)), color in zip(results.items(), colors):
    plt.plot(evals, hist, label=name, color=color)
plt.yscale("log")
plt.xlabel("Fitness evaluations")
plt.ylabel("Best fitness found (log scale)")
plt.title(f"Comparison of 14 Algorithms on Rastrigin Function (n_dim={N_DIM})")
plt.legend(ncol=2, fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_14_rastrigin.png", dpi=150)
plt.show()
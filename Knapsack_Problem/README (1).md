# Knapsack adaptations

These modules preserve the evaluation/return structure of the Rastrigin versions:
- `evaluation_count`
- `evaluation_history`
- `run_*` entry points
- best-history and position-history outputs
- `seed` and `verbose` parameters

Problem:
- 10 binary decisions
- items = [(2,3),(3,4),(4,5),(5,6),(9,10),(7,8),(1,2),(6,7),(8,9),(3,5)]
- capacity = 20
- objective is maximization of total value
- infeasible solutions have objective value 0

Important:
For algorithms originally written for continuous domains (ABC, ACOR, AFSA, AIS, BFOA, FA, GWO, IWD, IWO, PFA, PSO, WOA), the internal search remains continuous in [0,1]^n and is decoded to a binary knapsack vector using threshold 0.5 before fitness evaluation. This is a domain adaptation, not a claim that these are canonical binary versions of the algorithms.

GA and cGA use native binary representations.

The runner should use the returned `evaluation_history[-1]` as the actual evaluation count because several algorithms perform additional evaluations inside their iteration.

import numpy as np
import matplotlib.pyplot as plt


evaluation_count = 0


def evaluate(x):
    global evaluation_count
    evaluation_count += 1
    return sphere(x)


def sphere(x):
    return np.sum(x ** 2)


def decode(bitstring, n_dim, bits_per_var, bounds):
    lo, hi = bounds
    max_int = 2 ** bits_per_var - 1
    x = np.empty(n_dim)
    for i in range(n_dim):
        chunk = bitstring[i * bits_per_var:(i + 1) * bits_per_var]
        as_int = int("".join(chunk.astype(str)), 2)
        x[i] = lo + (as_int / max_int) * (hi - lo)
    return x


def fitness_of_bitstring(bitstring, n_dim, bits_per_var, bounds):
    x = decode(bitstring, n_dim, bits_per_var, bounds)
    return evaluate(x)


def run_cga(
    n_dim=10,
    bits_per_var=16,
    bounds=(-5.12, 5.12),
    pop_size=200,
    n_generations=3000,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    if seed is not None:
        np.random.seed(seed)

    genome_len = n_dim * bits_per_var
    prob_vector = np.full(genome_len, 0.5)
    
    best_history = []
    best_position_history = []
    best_ever_x = None
    best_ever_f = np.inf

    step = 1.0 / pop_size

    for gen in range(n_generations):
        ind1 = (np.random.rand(genome_len) < prob_vector).astype(int)
        ind2 = (np.random.rand(genome_len) < prob_vector).astype(int)

        f1 = fitness_of_bitstring(ind1, n_dim, bits_per_var, bounds)
        f2 = fitness_of_bitstring(ind2, n_dim, bits_per_var, bounds)

        if f1 <= f2:
            winner, loser, f_win = ind1, ind2, f1
        else:
            winner, loser, f_win = ind2, ind1, f2

        if f_win < best_ever_f:
            best_ever_f = f_win
            best_ever_x = decode(winner, n_dim, bits_per_var, bounds)

        diff_loci = winner != loser
        prob_vector[diff_loci & (winner == 1)] += step
        prob_vector[diff_loci & (winner == 0)] -= step
        prob_vector = np.clip(prob_vector, 0.0, 1.0)

        best_history.append(best_ever_f)
        best_position_history.append(best_ever_x.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 300 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:5d} | Best-ever: {best_ever_f:.6f}")

        if np.all((prob_vector == 0.0) | (prob_vector == 1.0)):
            if verbose:
                print(f"Probability vector converged at generation {gen}")
            break

    run_cga.evaluation_history = evaluation_history
    return best_ever_x, best_ever_f, best_history, best_position_history


if __name__ == "__main__":
    best_x, best_f, history, best_pos_history = run_cga(
        n_dim=10,
        bits_per_var=16,
        pop_size=100,
        n_generations=8000,
        seed=42,
    )

    print("\n=== Result (cGA) ===")
    print("Best solution x*:", np.round(best_x, 4))
    print("Best fitness f(x*):", best_f)
    print("Total fitness evaluations:", 2 * len(history))

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel("Best-ever fitness (log scale)")
    plt.title("cGA Convergence on Sphere Function")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

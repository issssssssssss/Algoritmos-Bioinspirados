import numpy as np
import matplotlib.pyplot as plt

from TSP_Problem import generate_cities, distance_matrix, tour_length


evaluation_count = 0


def decode_keys(bitstring, n_cities, bits_per_key):
    max_int = 2 ** bits_per_key - 1
    keys = np.empty(n_cities)
    for i in range(n_cities):
        chunk = bitstring[i * bits_per_key:(i + 1) * bits_per_key]
        as_int = int("".join(chunk.astype(str)), 2)
        keys[i] = as_int / max_int
    return np.argsort(keys)


def fitness_of_bitstring(bitstring, n_cities, bits_per_key, dist_matrix):
    global evaluation_count
    tour = decode_keys(bitstring, n_cities, bits_per_key)
    evaluation_count += 1
    return tour_length(tour, dist_matrix)


def run_cga(
    dist_matrix,
    n_cities,
    bits_per_key=16,
    pop_size=100,
    n_generations=15000,
    seed=None,
    verbose=True,
):
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    if seed is not None:
        np.random.seed(seed)

    genome_len = n_cities * bits_per_key
    prob_vector = np.full(genome_len, 0.5)

    best_history = []
    best_ever_tour = None
    best_ever_f = np.inf

    step = 1.0 / pop_size

    for gen in range(n_generations):
        ind1 = (np.random.rand(genome_len) < prob_vector).astype(int)
        ind2 = (np.random.rand(genome_len) < prob_vector).astype(int)

        f1 = fitness_of_bitstring(ind1, n_cities, bits_per_key, dist_matrix)
        f2 = fitness_of_bitstring(ind2, n_cities, bits_per_key, dist_matrix)

        if f1 <= f2:
            winner, loser, f_win = ind1, ind2, f1
        else:
            winner, loser, f_win = ind2, ind1, f2

        if f_win < best_ever_f:
            best_ever_f = f_win
            best_ever_tour = decode_keys(winner, n_cities, bits_per_key)

        diff_loci = winner != loser
        prob_vector[diff_loci & (winner == 1)] += step
        prob_vector[diff_loci & (winner == 0)] -= step
        prob_vector = np.clip(prob_vector, 0.0, 1.0)

        best_history.append(best_ever_f)
        evaluation_history.append(evaluation_count)

        if verbose and (gen % 500 == 0 or gen == n_generations - 1):
            print(f"Gen {gen:5d} | Best-ever tour length: {best_ever_f:.3f}")

        if np.all((prob_vector == 0.0) | (prob_vector == 1.0)):
            if verbose:
                print(f"Probability vector converged at generation {gen}")
            break

    run_cga.evaluation_history = evaluation_history
    return best_ever_tour, best_ever_f, best_history


if __name__ == "__main__":
    N_CITIES = 25
    cities = generate_cities(n_cities=N_CITIES, seed=42)
    dmat = distance_matrix(cities)

    best_tour, best_len, history = run_cga(
        dmat, N_CITIES, bits_per_key=16, pop_size=100,
        n_generations=15000, seed=42,
    )

    print("\n=== Result (cGA on TSP) ===")
    print("Best tour:", best_tour)
    print("Best tour length:", best_len)
    print("Total fitness evaluations:", 2 * len(history))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best-ever tour length")
    axes[0].set_title("cGA Convergence on TSP")
    axes[0].grid(True, alpha=0.3)

    tour_coords = cities[np.append(best_tour, best_tour[0])]
    axes[1].plot(tour_coords[:, 0], tour_coords[:, 1], "o-")
    for i, (x, y) in enumerate(cities):
        axes[1].annotate(str(i), (x, y), fontsize=8)
    axes[1].set_title(f"Best Tour Found (length={best_len:.2f})")

    plt.tight_layout()
    plt.show()
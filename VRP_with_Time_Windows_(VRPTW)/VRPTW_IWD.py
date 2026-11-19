import numpy as np
import matplotlib.pyplot as plt

from VRPTW_Problem import generate_instance, evaluate_solution, plot_routes, get_solution_details


evaluation_count = 0


def evaluate(solution, instance):
    global evaluation_count
    evaluation_count += 1
    return evaluate_solution(solution, instance)


def _construct_tour(soil, dmat, n_customers, rng, init_vel, a_v, b_v, c_v,
                     a_s, b_s, c_s, rho_iwd, epsilon_s):
    """
    Construye un giant tour con una gota de agua: en cada paso elige el
    siguiente cliente con probabilidad inversamente proporcional al
    suelo de la arista (menos suelo = más atractiva, igual que menos
    feromona residual en ACO pero en sentido opuesto). Actualiza la
    velocidad de la gota y remueve suelo localmente en cada arista que
    recorre, acumulando el suelo total removido (`drop_soil`), que se
    usa después para el refuerzo global de la mejor gota de la iteración.
    """
    unvisited = list(range(1, n_customers + 1))
    current = 0
    tour = []
    edges = []

    vel = init_vel
    drop_soil = 0.0

    while unvisited:
        soils = np.array([soil[current, j] for j in unvisited])
        g = soils - soils.min()  # >= 0, 0 en la arista con menos suelo

        # Normalizar por el rango observado en este paso antes de invertir:
        # con la fórmula clásica 1/(epsilon_s + g) y un epsilon_s absoluto
        # pequeño, basta con que UNA arista tenga algo menos de suelo que
        # las demás para que la probabilidad colapse a un argmin
        # determinista (ver nota en el docstring del módulo). Normalizar
        # g a una escala relativa ~[0, 1] antes de aplicar epsilon_s hace
        # que la selección sea invariante a la escala de la instancia y
        # mantiene exploración real en vez de un argmin encubierto.
        g_range = g.max()
        g_normalized = g / g_range if g_range > 1e-12 else g
        f = 1.0 / (epsilon_s + g_normalized)
        probs = f / f.sum()

        next_idx = rng.choice(len(unvisited), p=probs)
        next_city = unvisited.pop(next_idx)

        edge_soil = soil[current, next_city]
        vel = vel + a_v / (b_v + c_v * edge_soil ** 2)

        dist = dmat[current, next_city]
        time = dist / max(vel, 1e-6)
        delta_soil = a_s / (b_s + c_s * time ** 2)

        new_soil = (1 - rho_iwd) * soil[current, next_city] - rho_iwd * delta_soil
        soil[current, next_city] = new_soil
        soil[next_city, current] = new_soil
        drop_soil += delta_soil

        edges.append((current, next_city))
        tour.append(next_city)
        current = next_city

    return np.array(tour), drop_soil, edges


def run_iwd(
    instance,
    n_customers,
    n_drops=30,
    n_iterations=1000,
    init_vel=200.0,
    init_soil=10000.0,
    a_v=1000.0,
    b_v=0.01,
    c_v=1.0,
    a_s=1.0,
    b_s=0.01,
    c_s=1.0,
    rho_iwd=0.9,
    rho_n=0.9,
    epsilon_s=0.1,
    seed=None,
    verbose=True,
):
    """
    Intelligent Water Drops (Shah-Hosseini, 2009), adaptado a la
    representación de giant tour + split usada por el resto de los
    algoritmos. Cada iteración, `n_drops` gotas construyen un tour cada
    una; la mejor gota de la iteración refuerza (reduce aún más) el
    suelo de las aristas que usó, haciendo esa ruta más atractiva en
    iteraciones futuras -- el mecanismo de aprendizaje colectivo
    equivalente al depósito de feromona en ACO, pero con el signo
    invertido (aquí gana quien tiene MENOS suelo).
    """
    global evaluation_count, evaluation_history

    evaluation_count = 0
    evaluation_history = []

    rng = np.random.RandomState(seed)

    n_nodes = n_customers + 1  # depósito + clientes
    dmat = instance.dist_matrix
    soil = np.full((n_nodes, n_nodes), init_soil)
    np.fill_diagonal(soil, 0.0)

    global_best_solution = None
    global_best_fit = np.inf

    best_history = []
    mean_history = []
    best_position_history = []

    for it in range(n_iterations):
        drops_tours = []
        drops_soil = []
        drops_edges = []

        for _ in range(n_drops):
            tour, drop_soil, edges = _construct_tour(
                soil, dmat, n_customers, rng, init_vel,
                a_v, b_v, c_v, a_s, b_s, c_s, rho_iwd, epsilon_s,
            )
            drops_tours.append(tour)
            drops_soil.append(drop_soil)
            drops_edges.append(edges)

        fitness = np.array([evaluate(tour, instance) for tour in drops_tours])

        iter_best_idx = np.argmin(fitness)
        iter_best_edges = drops_edges[iter_best_idx]
        iter_best_drop_soil = drops_soil[iter_best_idx]

        # --- Refuerzo global: reducir suelo en las aristas de la mejor gota ---
        n_edges = max(1, len(iter_best_edges))
        for (a, b) in iter_best_edges:
            delta = rho_n * (1.0 / n_edges) * iter_best_drop_soil
            new_soil = (1 + rho_n) * soil[a, b] - delta
            soil[a, b] = new_soil
            soil[b, a] = new_soil

        if fitness[iter_best_idx] < global_best_fit:
            global_best_fit = fitness[iter_best_idx]
            global_best_solution = drops_tours[iter_best_idx].copy()

        best_history.append(global_best_fit)
        mean_history.append(np.mean(fitness))
        best_position_history.append(global_best_solution.copy())
        evaluation_history.append(evaluation_count)

        if verbose and (it % 50 == 0 or it == n_iterations - 1):
            print(f"Iter {it:4d} | Best cost: {global_best_fit:.3f}")

    run_iwd.evaluation_history = evaluation_history
    return global_best_solution, global_best_fit, best_history, mean_history, best_position_history


if __name__ == "__main__":
    N_CUSTOMERS = 25
    instance = generate_instance(n_customers=N_CUSTOMERS, seed=42)

    best_solution, best_cost, history, mean_history, position_history = run_iwd(
        instance, N_CUSTOMERS, n_drops=30, n_iterations=1000, seed=42,
    )

    routes, total_distance, n_vehicles = get_solution_details(best_solution, instance)

    print("\n=== Result (IWD on VRPTW) ===")
    print("Best solution (giant tour):", best_solution)
    print("Best cost:", best_cost)
    print("Distance:", total_distance, "| Vehicles used:", n_vehicles)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history, label="Best")
    axes[0].plot(mean_history, label="Mean", alpha=0.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].set_title("IWD Convergence on VRPTW")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_routes(instance, best_solution, ax=axes[1], title=f"Best Routes (cost={best_cost:.2f})")

    plt.tight_layout()
    plt.show()

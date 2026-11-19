"""
VRPTW_Problem.py

Utilidades compartidas para el Vehicle Routing Problem with Time Windows (VRPTW).
Todos los algoritmos (GA, PSO, GWO, WOA, ABC, ACO, ACOR) importan de aquí:

- generate_instance(...)      -> instancia sintética
- load_solomon_instance(...)  -> instancia desde archivo formato Solomon
- distance_matrix(...)
- decode_giant_tour(...)      -> convierte una permutación de clientes en rutas factibles
- evaluate_solution(...)      -> función de costo (fitness) usada por todos los algoritmos
- random_key_to_permutation() -> decodificación random-key para algoritmos de espacio continuo
- plot_routes(...)            -> visualización de la solución

Representación de solución (misma para los 7 algoritmos):
    Un "giant tour": permutación de los índices de clientes (1..n_customers).
    El depósito es el índice 0 y nunca aparece en la permutación.
    decode_giant_tour() parte esa permutación en varias rutas (vehículos)
    respetando capacidad y ventanas de tiempo. Los algoritmos de espacio
    continuo (PSO, GWO, WOA, ABC, ACOR) generan un vector de claves
    aleatorias (random keys) que se convierte en permutación vía argsort;
    GA y ACO operan directamente sobre la permutación.
"""

import numpy as np


# ============================================================
# CONTENEDOR DE INSTANCIA
# ============================================================

class VRPTWInstance:
    def __init__(self, coords, demands, time_windows, service_times,
                 vehicle_capacity, name="synthetic"):
        # coords[0] es el depósito
        self.coords = np.asarray(coords, dtype=float)
        self.demands = np.asarray(demands, dtype=float)
        self.time_windows = np.asarray(time_windows, dtype=float)  # (n+1, 2)
        self.service_times = np.asarray(service_times, dtype=float)
        self.vehicle_capacity = float(vehicle_capacity)
        self.name = name
        self.n_customers = len(self.coords) - 1
        self.dist_matrix = distance_matrix(self.coords)

    @property
    def depot_due(self):
        return self.time_windows[0, 1]

    def __repr__(self):
        return (f"VRPTWInstance(name={self.name!r}, n_customers={self.n_customers}, "
                f"capacity={self.vehicle_capacity})")


# ============================================================
# MATRIZ DE DISTANCIAS
# ============================================================

def distance_matrix(coords):
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ============================================================
# GENERADOR SINTÉTICO
# ============================================================

def generate_instance(
    n_customers=25,
    vehicle_capacity=100,
    area_size=100,
    demand_range=(5, 25),
    time_horizon=240,
    tw_width_range=(20, 60),
    service_time=10,
    seed=None,
    name=None,
):
    """
    Genera una instancia VRPTW sintética:
    - depósito en el centro del área
    - clientes con coordenadas uniformes
    - demandas enteras uniformes en demand_range
    - ventanas de tiempo factibles: siempre es posible llegar desde el
      depósito dentro del horizonte, con un ancho de ventana aleatorio.
    """
    rng = np.random.RandomState(seed)

    depot_coord = np.array([[area_size / 2, area_size / 2]])
    customer_coords = rng.uniform(0, area_size, size=(n_customers, 2))
    coords = np.vstack([depot_coord, customer_coords])

    demands = np.zeros(n_customers + 1)
    demands[1:] = rng.randint(demand_range[0], demand_range[1] + 1, size=n_customers)

    dmat = distance_matrix(coords)

    time_windows = np.zeros((n_customers + 1, 2))
    time_windows[0] = [0, time_horizon]

    for i in range(1, n_customers + 1):
        earliest_reachable = dmat[0, i]
        width = rng.uniform(tw_width_range[0], tw_width_range[1])
        latest_start = max(earliest_reachable + 1.0, time_horizon - width)
        latest_start = min(latest_start, time_horizon - 1e-6)
        ready = rng.uniform(earliest_reachable, latest_start)
        due = min(time_horizon, ready + width)
        time_windows[i] = [ready, due]

    service_times = np.full(n_customers + 1, float(service_time))
    service_times[0] = 0.0

    return VRPTWInstance(
        coords, demands, time_windows, service_times, vehicle_capacity,
        name=name or f"synthetic_n{n_customers}_seed{seed}",
    )


# ============================================================
# CARGA DE INSTANCIAS SOLOMON (formato estándar C1/C2/R1/R2/RC1/RC2)
# ============================================================

def load_solomon_instance(filepath):
    """
    Lee un archivo en formato Solomon estándar:

        <nombre>

        VEHICLE
        NUMBER     CAPACITY
          25         200

        CUSTOMER
        CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME

            0      40        50        0         0        1236        0
            1      45        68       10       912         967       90
            ...

    Devuelve un VRPTWInstance con la misma interfaz que generate_instance().
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    name = lines[0]

    veh_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("VEHICLE"))
    number_capacity_line = lines[veh_idx + 2].split()
    vehicle_capacity = float(number_capacity_line[1])

    cust_idx = next(i for i, l in enumerate(lines) if l.upper().startswith("CUSTOMER"))
    data_start = cust_idx + 2  # salta la línea de encabezado "CUST NO. XCOORD. ..."

    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        rows.append([float(x) for x in parts[:7]])

    rows = np.array(rows)
    rows = rows[np.argsort(rows[:, 0])]  # ordenar por CUST NO. (el depósito = 0 primero)

    coords = rows[:, 1:3]
    demands = rows[:, 3]
    ready = rows[:, 4]
    due = rows[:, 5]
    service = rows[:, 6]

    time_windows = np.stack([ready, due], axis=1)

    instance = VRPTWInstance(
        coords, demands, time_windows, service, vehicle_capacity, name=name
    )
    instance.n_vehicles_suggested = int(number_capacity_line[0])
    return instance


# ============================================================
# CARGA VÍA vrplib (pip install vrplib)
# ============================================================

def load_from_vrplib(filepath, instance_format="solomon"):
    """
    Alternativa a load_solomon_instance() que usa la librería `vrplib`
    (https://pypi.org/project/vrplib/) para leer el archivo. Útil si
    prefieres apoyarte en un parser mantenido externamente, o si más
    adelante quieres leer instancias en formato "vrplib"/CVRPLIB
    (instance_format="vrplib") además de instancias Solomon/Homberger-
    Gehring (instance_format="solomon").

    Nota: la versión actual de `vrplib` (2.2.0) ya NO incluye las
    funciones `download_instance` / `list_names` que existían en
    versiones antiguas -- hay que descargar el archivo .txt a mano
    (p. ej. desde SINTEF) y pasarle la ruta local a esta función.

    Requiere: pip install vrplib
    """
    try:
        import vrplib
    except ImportError:
        raise ImportError(
            "load_from_vrplib requiere el paquete vrplib. "
            "Instálalo con: pip install vrplib"
        )

    data = vrplib.read_instance(filepath, instance_format=instance_format)

    instance = VRPTWInstance(
        coords=data["node_coord"],
        demands=data["demand"],
        time_windows=data["time_window"],
        service_times=data["service_time"],
        vehicle_capacity=data["capacity"],
        name=data.get("name", filepath),
    )
    if "vehicles" in data:
        instance.n_vehicles_suggested = int(data["vehicles"])

    return instance


# ============================================================
# DECODIFICACIÓN: GIANT TOUR -> RUTAS FACTIBLES
# ============================================================

def decode_giant_tour(permutation, instance, vehicle_capacity=None):
    """
    Divide un "giant tour" (permutación de índices de clientes 1..n) en
    rutas factibles mediante un split voraz: se agrega el siguiente
    cliente a la ruta actual si no se viola la capacidad ni la ventana
    de tiempo (llegar antes de due_time; esperar si se llega antes de
    ready_time). Si se violaría alguna, se cierra la ruta actual y se
    inicia una nueva desde el depósito.

    Devuelve: lista de rutas, cada una una lista de índices de clientes
    (el depósito no se incluye explícitamente, se asume al inicio/fin).
    """
    capacity = vehicle_capacity if vehicle_capacity is not None else instance.vehicle_capacity
    dmat = instance.dist_matrix
    tw = instance.time_windows
    service = instance.service_times
    demands = instance.demands

    routes = []
    current_route = []
    current_load = 0.0
    current_time = 0.0
    current_pos = 0  # depósito

    for cust in permutation:
        cust = int(cust)
        demand = demands[cust]
        arrival = current_time + dmat[current_pos, cust]
        start_service = max(arrival, tw[cust, 0])

        capacity_ok = (current_load + demand) <= capacity + 1e-9
        time_ok = start_service <= tw[cust, 1] + 1e-9

        if current_route and (not capacity_ok or not time_ok):
            routes.append(current_route)
            current_route = []
            current_load = 0.0
            current_time = 0.0
            current_pos = 0
            arrival = dmat[0, cust]
            start_service = max(arrival, tw[cust, 0])

        current_route.append(cust)
        current_load += demand
        current_time = start_service + service[cust]
        current_pos = cust

    if current_route:
        routes.append(current_route)

    return routes


# ============================================================
# COSTO Y PENALIZACIONES DE UNA RUTA
# ============================================================

def route_cost_and_penalty(route, instance):
    dmat = instance.dist_matrix
    tw = instance.time_windows
    service = instance.service_times
    demands = instance.demands
    capacity = instance.vehicle_capacity

    distance = 0.0
    time_penalty = 0.0
    load = 0.0
    t = 0.0
    pos = 0

    for cust in route:
        distance += dmat[pos, cust]
        arrival = t + dmat[pos, cust]
        start = max(arrival, tw[cust, 0])
        if start > tw[cust, 1]:
            time_penalty += (start - tw[cust, 1])
        t = start + service[cust]
        load += demands[cust]
        pos = cust

    distance += dmat[pos, 0]  # regreso al depósito
    capacity_penalty = max(0.0, load - capacity)

    return distance, time_penalty, capacity_penalty


# ============================================================
# FUNCIÓN DE EVALUACIÓN (FITNESS) USADA POR TODOS LOS ALGORITMOS
# ============================================================

def evaluate_solution(
    permutation,
    instance,
    vehicle_penalty=0.0,
    time_window_penalty_weight=1000.0,
    capacity_penalty_weight=1000.0,
):
    """
    Costo total = distancia recorrida
                + vehicle_penalty * n_vehiculos_usados
                + time_window_penalty_weight * violación_total_de_ventanas
                + capacity_penalty_weight * exceso_total_de_capacidad

    La capacidad casi siempre queda satisfecha por construcción en
    decode_giant_tour, pero se deja el término de penalización por
    robustez (p. ej. si se pasa una vehicle_capacity distinta).
    """
    routes = decode_giant_tour(permutation, instance)

    total_distance = 0.0
    total_time_penalty = 0.0
    total_capacity_penalty = 0.0

    for route in routes:
        d, tp, cp = route_cost_and_penalty(route, instance)
        total_distance += d
        total_time_penalty += tp
        total_capacity_penalty += cp

    n_vehicles = len(routes)

    cost = (
        total_distance
        + vehicle_penalty * n_vehicles
        + time_window_penalty_weight * total_time_penalty
        + capacity_penalty_weight * total_capacity_penalty
    )

    return cost


def get_solution_details(permutation, instance):
    """Devuelve (rutas, distancia_total, n_vehiculos) sin penalizaciones,
    útil para reportar resultados finales."""
    routes = decode_giant_tour(permutation, instance)
    total_distance = sum(route_cost_and_penalty(r, instance)[0] for r in routes)
    return routes, total_distance, len(routes)


# ============================================================
# CODIFICACIÓN RANDOM-KEY (para algoritmos de espacio continuo)
# ============================================================

def random_key_to_permutation(keys):
    """
    Convierte un vector continuo de longitud n_customers en una
    permutación de clientes (índices 1..n_customers) vía argsort.
    """
    order = np.argsort(keys)
    return order + 1  # +1 porque el índice 0 está reservado al depósito


# ============================================================
# VISUALIZACIÓN
# ============================================================

def plot_routes(instance, permutation, ax=None, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    routes = decode_giant_tour(permutation, instance)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    coords = instance.coords
    n_routes = max(len(routes), 1)
    colors = cm.tab20(np.linspace(0, 1, n_routes))

    ax.scatter(coords[0, 0], coords[0, 1], c="black", marker="s", s=100, zorder=5, label="Depósito")
    ax.scatter(coords[1:, 0], coords[1:, 1], c="gray", s=20, zorder=3)

    for i, route in enumerate(routes):
        seq = [0] + route + [0]
        pts = coords[seq]
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=colors[i], markersize=4, linewidth=1.5,
                 label=f"Ruta {i + 1}" if len(routes) <= 10 else None)

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=7)
    return ax
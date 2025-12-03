"""
route_engine.py

Core routing logic for multi-stop route optimization.
Supports BOTH:
    (1) Address-based routing (default)
    (2) Node-based routing (required for figures and ML)
"""

import itertools
import osmnx as ox

from preprocessing import address_to_node
from config import START_ADDRESS, DEBUG


# =======================================================
# Cost Matrix Computation
# =======================================================

def _compute_cost_matrix(G, nodes, weight):
    """
    Compute pairwise cost matrix between nodes using a given edge weight.
    weight: 'length' OR 'travel_time'
    """
    n = len(nodes)
    M = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            try:
                path = ox.shortest_path(G, nodes[i], nodes[j], weight=weight)
                if path is None:
                    M[i][j] = float("inf")
                    continue

                attrs = ox.utils_graph.get_route_edge_attributes(G, path, weight)
                M[i][j] = sum(attrs)

            except Exception:
                M[i][j] = float("inf")

    return M


# =======================================================
# Brute Force Multi-stop Route Search
# =======================================================

def _brute_force_best_route(cost_matrix, start_idx, stop_indices, end_idx):
    """
    Solve small multi-stop ordering via brute force.
    Returns: (best_route_index_tuple, best_cost)
    """
    best_route = None
    best_cost = float("inf")

    for perm in itertools.permutations(stop_indices):
        route = (start_idx,) + perm + (end_idx,)
        cost = sum(cost_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost


# =======================================================
# Main Routing Function
# =======================================================

def plan_multi_stop_route(
    G,
    start_address=None,
    stop_addresses=None,
    end_address=None,
    preference="distance",
    start_node=None,
    stop_nodes=None,
    end_node=None,
):
    """
    Multi-stop route optimizer.

    Supports two modes:
    -------------------------------------
    1. Address mode (original behavior)
       start_address="701...", stop_addresses=[...], end_address="..."

    2. Node mode (for plotting and scripts using raw lat/lon)
       start_node=12345, stop_nodes=[111, 222], end_node=9999
    -------------------------------------
    """

    # ======================================================
    # MODE 1: Node-based routing (NO geocoding required)
    # ======================================================
    if start_node is not None and stop_nodes is not None and end_node is not None:
        all_nodes = [start_node] + stop_nodes + [end_node]

    # ======================================================
    # MODE 2: Address-based routing (fallback to original)
    # ======================================================
    else:
        # Import here to avoid circular import issues
        from preprocessing import address_to_node

        # Convert addresses → nodes
        start_node, _, _ = address_to_node(G, start_address)
        stop_nodes = [address_to_node(G, addr)[0] for addr in stop_addresses]
        end_node, _, _ = address_to_node(G, end_address)

        all_nodes = [start_node] + stop_nodes + [end_node]

    # ======================================================
    # Build cost matrix (distance or time)
    # ======================================================
    import networkx as nx
    import itertools

    n = len(all_nodes)

    # Build adjacency matrix for the chosen metric
    weight = "length" if preference == "distance" else "travel_time"

    cost_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    path = nx.shortest_path(G, all_nodes[i], all_nodes[j], weight=weight)
                    cost = sum(G[u][v][0][weight] for u, v in zip(path[:-1], path[1:]))
                except Exception:
                    cost = float("inf")
                cost_matrix[i][j] = cost

    # ======================================================
    # Solve mini-TSP via brute force
    # ======================================================
    start_idx = 0
    stop_indices = list(range(1, n - 1))
    end_idx = n - 1

    best_cost = float("inf")
    best_route_idx = None

    for perm in itertools.permutations(stop_indices):
        candidate = [start_idx] + list(perm) + [end_idx]

        total = sum(
            cost_matrix[candidate[i]][candidate[i + 1]]
            for i in range(len(candidate) - 1)
        )

        if total < best_cost:
            best_cost = total
            best_route_idx = candidate

    # Final route (as node IDs)
    route_nodes = [all_nodes[i] for i in best_route_idx]
    route_latlons = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_nodes]

    return {
        "nodes": route_nodes,
        "latlons": route_latlons,
        "total_cost": best_cost,
        "preference": preference,
    }


"""
preprocessing.py

Utilities for working with addresses, coordinates, and the road graph.
"""

import os
import osmnx as ox

from config import (
    START_ADDRESS,
    GRAPH_DIST_METERS,
    GRAPHML_PATH,
    NETWORK_TYPE,
    DEBUG,
)


def load_graph():
    """
    Load a cached driving graph if it exists; otherwise download and cache it.
    Uses a circle around START_ADDRESS with radius GRAPH_DIST_METERS.
    """
    os.makedirs(os.path.dirname(GRAPHML_PATH), exist_ok=True)

    if os.path.exists(GRAPHML_PATH):
        if DEBUG:
            print(f"[preprocessing] Loading graph from {GRAPHML_PATH}")
        G = ox.load_graphml(GRAPHML_PATH)
    else:
        if DEBUG:
            print("[preprocessing] Downloading graph from OSM...")
        lat, lon = ox.geocode(START_ADDRESS)
        G = ox.graph_from_point(
            (lat, lon), dist=GRAPH_DIST_METERS, network_type=NETWORK_TYPE
        )

        # Keep largest connected component
        try:
            G = ox.utils_graph.get_largest_component(G)
        except AttributeError:
            # For older osmnx versions
            G = ox.utils_graph.largest_component(G)

        # Add speed + travel time attributes
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        ox.save_graphml(G, GRAPHML_PATH)
        if DEBUG:
            print(f"[preprocessing] Saved graph to {GRAPHML_PATH}")

    return G


def geocode_address(address: str):
    """
    Use OSM/Nominatim via OSMnx to convert an address to (lat, lon).
    """
    lat, lon = ox.geocode(address)
    return lat, lon


def nearest_node(G, lat: float, lon: float):
    """
    Find nearest drivable OSM node in graph G given a latitude and longitude.
    """
    node_id = ox.nearest_nodes(G, lon, lat)
    return node_id


def address_to_node(G, address: str):
    """
    Shortcut: address → (lat, lon) → nearest node.
    Returns: (node_id, lat, lon)
    """
    lat, lon = geocode_address(address)
    node_id = nearest_node(G, lat, lon)
    return node_id, lat, lon

FALLBACK_POI = {
    "ann arbor public library": (42.279706, -83.748913),
    "ann arbor district library": (42.279706, -83.748913),
    "downtown library": (42.279706, -83.748913),
}


def geocode_address(address: str):
    address_norm = address.lower().strip()
    if address_norm in FALLBACK_POI:
        return FALLBACK_POI[address_norm]

    try:
        lat, lon = ox.geocode(address)
        return lat, lon
    except Exception:
        raise RuntimeError(f"Could not geocode address: {address}")

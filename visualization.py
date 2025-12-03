"""
visualization.py

Helper functions to visualize routes on the road network and
generate Google Maps URLs.
"""

import os
import osmnx as ox
import matplotlib.pyplot as plt


def plot_route(G, route_nodes, save_path=None, show=True):
    """
    Plot a route (node sequence) on the graph G.
    """
    fig, ax = ox.plot_graph_route(G, route_nodes, show=False, close=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[visualization] Saved route plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_google_maps_link(latlons):
    """
    Given a list of (lat, lon) pairs in order,
    return a Google Maps directions URL.
    """
    base = "https://www.google.com/maps/dir/"
    segments = [f"{lat},{lon}" for lat, lon in latlons]
    return base + "/".join(segments)

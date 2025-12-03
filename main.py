"""
main.py

User interface for the carpool route planner.
Allows users to manually enter up to 5 intermediate stops.
"""

from preprocessing import load_graph, address_to_node
from route_engine import plan_multi_stop_route
from visualization import plot_route, generate_google_maps_link
from config import START_ADDRESS
from models import load_duration_model, recommend_ml_route

import webbrowser


def prompt_intermediate_stops(max_stops=5):
    """
    Ask the user for up to 5 intermediate stops.
    User may enter 0–5 stops by pressing Enter early.
    """
    stops = []
    print(f"\nEnter up to {max_stops} intermediate stops.")
    print("Press Enter with no input to finish.\n")

    for i in range(max_stops):
        inp = input(f"Enter stop #{i+1} (or press Enter to stop): ").strip()
        if inp == "":
            break
        stops.append(inp)

    print(f"\nIntermediate stops entered: {stops}\n")
    return stops


def prompt_user_inputs():
    """
    Ask user for start, end, intermediate stops, and preference.
    """
    print("=== Carpool Route Planner ===")
    print(f"(Default start address if blank: {START_ADDRESS})\n")

    start = input("Enter START address (or press Enter to use default): ").strip()
    end = input("Enter DESTINATION address: ").strip()

    # Intermediate stops
    stop_addrs = prompt_intermediate_stops(max_stops=5)

    # Mode selection
    print("\nOptimization preference:")
    print("  1) Save gas (shortest distance)")
    print("  2) Save time (shortest travel time)")
    print("  3) ML recommended route (learned from real travel times!)")
    choice = input("Choose 1, 2, or 3 [default=2]: ").strip()

    if choice == "1":
        preference = "distance"
    elif choice == "3":
        preference = "ml"
    else:
        preference = "time"

    return start, stop_addrs, end, preference


def main():
    # 1. Capture user inputs
    start_addr, stop_addrs, end_addr, pref = prompt_user_inputs()

    # Use default start if blank
    if not start_addr.strip():
        start_addr = START_ADDRESS

    # 2. Load graph (cached or downloaded)
    G = load_graph()

    # 3. Routing logic
    if pref in ("distance", "time"):
        # Classical routing
        result = plan_multi_stop_route(
            G,
            start_address=start_addr,
            stop_addresses=stop_addrs,
            end_address=end_addr,
            preference=pref,
        )
        route_nodes = result["nodes"]
        route_latlons = result["latlons"]

        print("\n=== Best Route (Classical Routing Engine) ===")
        print(f"Preference: {result['preference']}")
        print(f"Total cost: {result['total_cost']:.2f}")

    else:
        # ML-based recommended route
        print("\n[main] Loading ML duration model...")
        model = load_duration_model()

        # Convert address → graph nodes
        start_node, _, _ = address_to_node(G, start_addr)
        end_node, _, _ = address_to_node(G, end_addr)

        stop_nodes = []
        for addr in stop_addrs:
            sn, _, _ = address_to_node(G, addr)
            stop_nodes.append(sn)

        # Generate ML route recommendation
        print("[main] ML Ranking of Route Permutations...")
        result = recommend_ml_route(model, G, start_node, stop_nodes, end_node)
        route_nodes = result["nodes"]
        route_latlons = result["latlons"]

        print("\n=== ML Recommended Route ===")
        print(f"Predicted travel time: {result['predicted_duration']:.2f} seconds")

    # 4. Google Maps link export
    gmap_url = generate_google_maps_link(route_latlons)
    print("\nGoogle Maps Directions Link:")
    print(gmap_url)

    print("\nOpening in your default browser...")
    try:
        webbrowser.open(gmap_url)
    except:
        print("[main] Could not automatically open browser.")

    # 5. Pretty-print coordinates
    print("\nRoute sequence (lat, lon):")
    for lat, lon in route_latlons:
        print(f"  ({lat:.6f}, {lon:.6f})")

    # 6. Plot route on map + save
    try:
        plot_route(G, route_nodes, save_path="output/route.png", show=True)
    except Exception as e:
        print(f"[main] Could not plot route: {e}")


if __name__ == "__main__":
    main()

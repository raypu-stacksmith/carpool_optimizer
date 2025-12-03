"""
config.py

Global configuration for the carpool route planner.
"""

# Your default fixed starting point (can still be overridden by user input)
START_ADDRESS = "701 E University Ave, Ann Arbor, MI 48109"

# Default graph area: we'll download roads within this radius (in meters)
GRAPH_DIST_METERS = 3000

# Area name for OSMnx (just used if you want to change how the graph is built)
DEFAULT_CITY = "Ann Arbor, Michigan, USA"

# Where to cache the road network so we don't re-download it every time
GRAPHML_PATH = "data/ann_arbor_drive.graphml"

# Real route CSV and ML model paths (your real data!)
ROUTE_CSV_PATH = "data/ann_arbor_real_routes.csv"
DURATION_MODEL_PATH = "models/route_duration_mlp.pth"

# OSMnx network type
NETWORK_TYPE = "drive"   # you can change to "drive_service" or "walk" etc.

# Debug flag
DEBUG = True




import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import osmnx as ox
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import ROUTE_CSV_PATH, DURATION_MODEL_PATH


FEATURE_COLS = ["orig_lat", "orig_lon", "dest_lat", "dest_lon", "distance_m"]
TARGET_COL = "duration_s"


# -----------------------------------------------------
# MODEL DEFINITION
# -----------------------------------------------------
class RouteDurationMLP(nn.Module):
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
def load_real_route_data(csv_path=ROUTE_CSV_PATH):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values.astype("float32")
    y = df[TARGET_COL].values.astype("float32").reshape(-1, 1)
    return X, y, df


# -----------------------------------------------------
# TRAIN MODEL (with history)
# -----------------------------------------------------
def train_duration_model(
    epochs=80, batch_size=32, lr=1e-3, save_path=DURATION_MODEL_PATH
):
    X, y, _ = load_real_route_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RouteDurationMLP(input_dim=X_train_t.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {
        "epoch": [],
        "loss": [],
        "mae": [],
        "rmse": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for bx, by in loader:
            opt.zero_grad()
            preds = model(bx)
            loss = loss_fn(preds, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Eval on test
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).numpy()
            real = y_test
            mae = mean_absolute_error(real, preds)
            rmse = math.sqrt(mean_squared_error(real, preds))

        # Log metrics
        history["epoch"].append(epoch + 1)
        history["loss"].append(total_loss)
        history["mae"].append(mae)
        history["rmse"].append(rmse)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"Loss={total_loss:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}"
        )

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[models] Saved duration model to {save_path}")

    return model, history


# -----------------------------------------------------
# LOAD MODEL (ALWAYS returns model, history)
# -----------------------------------------------------
def load_duration_model(model_path=DURATION_MODEL_PATH):

    # Model exists: load it
    if os.path.exists(model_path):
        model = RouteDurationMLP(input_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        print(f"[models] Loaded duration model from {model_path}")
        return model, None

    # Otherwise train a new one
    print("[models] No saved model found. Training a new model...")
    model, history = train_duration_model(save_path=model_path)
    return model, history


# -----------------------------------------------------
# ML PREDICTION FOR A LEG
# -----------------------------------------------------
def _predict_leg_duration(model, G, n1, n2):
    nd1 = G.nodes[n1]
    nd2 = G.nodes[n2]

    orig_lat, orig_lon = nd1["y"], nd1["x"]
    dest_lat, dest_lon = nd2["y"], nd2["x"]

    # Great-circle distance for OSMnx 2.0.7
    dist = ox.distance.great_circle(orig_lat, orig_lon, dest_lat, dest_lon)

    fv = [orig_lat, orig_lon, dest_lat, dest_lon, dist]
    X = torch.tensor([fv], dtype=torch.float32)

    with torch.no_grad():
        return model(X).item()


# -----------------------------------------------------
# SCORE PERMUTATIONS
# -----------------------------------------------------
def score_routes(model, G, route_nodes_list):
    results = []
    for route in route_nodes_list:
        total = 0.0
        for i in range(len(route) - 1):
            total += _predict_leg_duration(model, G, route[i], route[i + 1])
        results.append((route, total))
    return results


# -----------------------------------------------------
# ML RECOMMENDED ROUTE
# -----------------------------------------------------
def recommend_ml_route(model, G, start_node, stop_nodes, end_node):
    import itertools

    all_routes = []
    for perm in itertools.permutations(stop_nodes):
        r = [start_node] + list(perm) + [end_node]
        all_routes.append(r)

    scored = score_routes(model, G, all_routes)
    best_route, best_pred = min(scored, key=lambda x: x[1])

    latlons = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in best_route]

    return {
        "nodes": best_route,
        "latlons": latlons,
        "predicted_duration": best_pred,
        "preference": "machine_learning",
    }



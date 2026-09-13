import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SIZES = [100, 200, 300, 400, 500]

FEATURES = [
    "energy",
    "trust",
    "drop_rate",
    "forward_ratio",
]

TARGET = "label"

EPOCHS = 10


# ============================================================
# Same 114-parameter 2-layer GCN required by the guide
# ============================================================

class StaticGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


# ============================================================
# Build undirected edge index
# ============================================================

def build_edge_index(edges):

    src = torch.tensor(
        edges["src"].to_numpy(),
        dtype=torch.long
    )

    dst = torch.tensor(
        edges["dst"].to_numpy(),
        dtype=torch.long
    )

    # Make graph undirected
    edge_index = torch.stack(
        [
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ],
        dim=0,
    )

    return edge_index


# ============================================================
# Load one dataset size
# ============================================================

def load_dataset(size):

    folder = DATA_DIR / str(size)

    nodes_path = folder / "nodes.csv"
    edges_path = folder / "edges.csv"

    print(f"\nLoading {size}-node dataset...")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    print(f"Nodes rows: {len(nodes):,}")
    print(f"Edges rows: {len(edges):,}")

    # --------------------------------------------------------
    # Use first snapshot for a clean scalability comparison
    # --------------------------------------------------------

    first_run = nodes["run"].min()
    first_time = nodes.loc[
        nodes["run"] == first_run,
        "time"
    ].min()

    node_snapshot = nodes[
        (nodes["run"] == first_run)
        & (nodes["time"] == first_time)
    ].copy()

    edge_snapshot = edges[
        (edges["run"] == first_run)
        & (edges["time"] == first_time)
    ].copy()

    node_snapshot = node_snapshot.sort_values("node_id")

    # --------------------------------------------------------
    # Map node IDs to 0...N-1
    # --------------------------------------------------------

    node_ids = node_snapshot["node_id"].to_numpy()

    node_map = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    edge_snapshot = edge_snapshot[
        edge_snapshot["src"].isin(node_map)
        & edge_snapshot["dst"].isin(node_map)
    ].copy()

    edge_snapshot["src"] = edge_snapshot["src"].map(node_map)
    edge_snapshot["dst"] = edge_snapshot["dst"].map(node_map)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    x_np = node_snapshot[FEATURES].to_numpy(dtype=np.float32)

    # Scale this snapshot only for the benchmark
    scaler = StandardScaler()
    x_np = scaler.fit_transform(x_np).astype(np.float32)

    x = torch.tensor(x_np, dtype=torch.float32)

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = torch.tensor(
        node_snapshot[TARGET].to_numpy(),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edge_index = build_edge_index(edge_snapshot)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )

    return data


# ============================================================
# Train and benchmark
# ============================================================

def benchmark(size, device):

    data = load_dataset(size)

    data = data.to(device)

    model = StaticGCN().to(device)

    # Confirm parameter count
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 114, (
        f"Expected 114 parameters, got {parameter_count}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    model.train()

    for _ in range(2):

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = F.cross_entropy(
            logits,
            data.y,
        )

        loss.backward()
        optimizer.step()

    # --------------------------------------------------------
    # Timed training
    # --------------------------------------------------------

    start = time.perf_counter()

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = F.cross_entropy(
            logits,
            data.y,
        )

        loss.backward()
        optimizer.step()

    # Synchronize GPU/MPS before stopping timer
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        predictions = logits.argmax(
            dim=1
        )

        accuracy = (
            predictions == data.y
        ).float().mean().item()

    return {
        "nodes": size,
        "parameters": parameter_count,
        "edges": data.edge_index.shape[1],
        "epochs": EPOCHS,
        "accuracy": accuracy,
        "time_seconds": elapsed,
    }


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if torch.cuda.is_available():

        device = torch.device("cuda")

    elif torch.backends.mps.is_available():

        device = torch.device("mps")

    else:

        device = torch.device("cpu")

    print("=" * 60)
    print("GNN SCALABILITY BENCHMARK")
    print("=" * 60)

    print(f"Using device: {device}")
    print(f"Node sizes: {SIZES}")
    print(f"Epochs per size: {EPOCHS}")

    results = []

    for size in SIZES:

        result = benchmark(
            size,
            device
        )

        results.append(result)

        print(
            f"{size:>3} nodes | "
            f"accuracy={result['accuracy']:.4f} | "
            f"time={result['time_seconds']:.4f}s | "
            f"parameters={result['parameters']}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_csv = RESULTS_DIR / "scalability.csv"

    results_df.to_csv(
        output_csv,
        index=False
    )

    print()
    print(f"Saved: {output_csv}")

    # ========================================================
    # Accuracy plot
    # ========================================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["nodes"],
        results_df["accuracy"],
        marker="o",
    )

    plt.xlabel("Number of nodes")
    plt.ylabel("Accuracy")
    plt.title("Static-GCN Scalability: Accuracy")

    plt.grid(True)

    accuracy_path = RESULTS_DIR / "scalability_accuracy.png"

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {accuracy_path}")

    # ========================================================
    # Runtime plot
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        results_df["nodes"],
        results_df["time_seconds"],
        marker="o",
    )

    plt.xlabel("Number of nodes")
    plt.ylabel("Training time (seconds)")
    plt.title("Static-GCN Scalability: Training Time")

    plt.grid(True)

    time_path = RESULTS_DIR / "scalability_time.png"

    plt.savefig(
        time_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {time_path}")


if __name__ == "__main__":
    main()

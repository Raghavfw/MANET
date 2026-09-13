import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES = [
    "drop_rate",
    "trust",
    "forward_ratio",
    "energy",
]

TARGET = "label"

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("results")

MODEL_NAMES = [
    "Static-GCN",
    "GraphSAGE",
    "GAT",
    "EvolveGCN-H",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, predictions):

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "mal_precision": precision_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "mal_recall": recall_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "mal_f1": f1_score(
            labels,
            predictions,
            pos_label=1,
            zero_division=0
        ),
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_temporal(times):

    unique_times = sorted(
        list(times)
    )

    split_index = int(
        len(unique_times) * 0.70
    )

    train_times = unique_times[
        :split_index
    ]

    test_times = unique_times[
        split_index:
    ]

    return train_times, test_times


# ============================================================
# EDGE INDEX
# ============================================================

def build_edge_index(
    edge_df,
    node_id_to_index,
    device
):

    src = []
    dst = []

    for row in edge_df.itertuples():

        if (
            row.src not in node_id_to_index
            or row.dst not in node_id_to_index
        ):
            continue

        src.append(
            node_id_to_index[row.src]
        )

        dst.append(
            node_id_to_index[row.dst]
        )

    if len(src) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=device
        )

        return edge_index

    src = torch.tensor(
        src,
        dtype=torch.long,
        device=device
    )

    dst = torch.tensor(
        dst,
        dtype=torch.long,
        device=device
    )

    # Make the graph undirected.
    edge_index = torch.cat(
        [
            torch.stack(
                [src, dst],
                dim=0
            ),

            torch.stack(
                [dst, src],
                dim=0
            ),
        ],
        dim=1
    )

    return edge_index


# ============================================================
# STATIC GCN
# ============================================================

class StaticGCN(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# GRAPHSAGE
# ============================================================

class GraphSAGE(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# GAT
# ============================================================

class GAT(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=1
        )

        self.conv2 = GATConv(
            hidden_channels,
            out_channels,
            heads=1
        )

    def forward(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.elu(x)

        x = self.conv2(
            x,
            edge_index
        )

        return x


# ============================================================
# CUSTOM EVOLVEGCN-H FALLBACK
#
# NOTE:
# The official torch_geometric_temporal package could not
# be installed because torch-sparse failed to build.
#
# This is therefore a custom EvolveGCN-H-style implementation,
# NOT the official torch_geometric_temporal EvolveGCNH class.
#
# The model maintains an evolving hidden GCN weight state
# across ordered graph snapshots.
# ============================================================

class EvolveGCNH(torch.nn.Module):

    def __init__(
        self,
        in_channels=4,
        hidden_channels=16,
        out_channels=2
    ):

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Initial GCN weight.
        self.initial_weight = torch.nn.Parameter(
            torch.empty(
                in_channels,
                hidden_channels
            )
        )

        torch.nn.init.xavier_uniform_(
            self.initial_weight
        )

        # Flattened first-layer weight size.
        weight_size = (
            in_channels *
            hidden_channels
        )

        # Encode graph information.
        self.graph_encoder = torch.nn.Linear(
            hidden_channels,
            weight_size
        )

        # Recurrently evolve the GCN weight.
        self.weight_gru = torch.nn.GRUCell(
            weight_size,
            weight_size
        )

        # Initial bias.
        self.initial_bias = torch.nn.Parameter(
            torch.zeros(
                hidden_channels
            )
        )

        # Output graph convolution.
        self.output_conv = GCNConv(
            hidden_channels,
            out_channels
        )

    def forward(
        self,
        x,
        edge_index,
        hidden=None
    ):

        # ----------------------------------------------------
        # 1. Select current GCN weight
        # ----------------------------------------------------

        if hidden is None:

            current_weight = (
                self.initial_weight
            )

        else:

            current_weight = hidden.view(
                self.in_channels,
                self.hidden_channels
            )

        # ----------------------------------------------------
        # 2. First graph transformation
        # ----------------------------------------------------

        graph_x = torch.matmul(
            x,
            current_weight
        )

        graph_x = (
            graph_x +
            self.initial_bias
        )

        graph_x = F.relu(
            graph_x
        )

        # ----------------------------------------------------
        # 3. Obtain graph-level representation
        # ----------------------------------------------------

        pooled = graph_x.mean(
            dim=0
        )

        # ----------------------------------------------------
        # 4. Encode graph information
        # ----------------------------------------------------

        gru_input = self.graph_encoder(
            pooled
        )

        # ----------------------------------------------------
        # 5. Evolve the GCN weight
        # ----------------------------------------------------

        if hidden is None:

            previous_state = (
                self.initial_weight
                .reshape(-1)
            )

        else:

            previous_state = hidden

        hidden = self.weight_gru(
            gru_input,
            previous_state
        )

        evolved_weight = hidden.view(
            self.in_channels,
            self.hidden_channels
        )

        # ----------------------------------------------------
        # 6. Apply evolved weight
        # ----------------------------------------------------

        x = torch.matmul(
            x,
            evolved_weight
        )

        x = x + self.initial_bias

        x = F.relu(x)

        # ----------------------------------------------------
        # 7. Final GCN classifier
        # ----------------------------------------------------

        out = self.output_conv(
            x,
            edge_index
        )

        return out, hidden


# ============================================================
# SNAPSHOT PREPARATION
# ============================================================

def load_data(
    data_dir,
    device
):

    nodes_path = (
        data_dir /
        "nodes_dynamic.csv"
    )

    edges_path = (
        data_dir /
        "edges_dynamic.csv"
    )

    print(
        f"Loading nodes from: {nodes_path}"
    )

    print(
        f"Loading edges from: {edges_path}"
    )

    nodes_df = pd.read_csv(
        nodes_path
    )

    edges_df = pd.read_csv(
        edges_path
    )

    return nodes_df, edges_df


# ============================================================
# SCALER
# ============================================================

def fit_scaler(
    nodes_df,
    train_times
):

    train_df = nodes_df[
        nodes_df["time"].isin(
            train_times
        )
    ]

    scaler = StandardScaler()

    scaler.fit(
        train_df[FEATURES]
    )

    return scaler


# ============================================================
# BUILD SNAPSHOTS
# ============================================================

def prepare_snapshots(
    nodes_df,
    edges_df,
    times,
    scaler,
    device
):

    snapshots = {}

    for time_s in times:

        node_snapshot = (
            nodes_df[
                nodes_df["time"] == time_s
            ]
            .sort_values("node_id")
            .copy()
        )

        node_ids = (
            node_snapshot[
                "node_id"
            ]
            .tolist()
        )

        node_id_to_index = {
            node_id: index
            for index, node_id
            in enumerate(node_ids)
        }

        x_np = scaler.transform(
            node_snapshot[
                FEATURES
            ]
        )

        x = torch.tensor(
            x_np,
            dtype=torch.float32,
            device=device
        )

        y = torch.tensor(
            node_snapshot[
                TARGET
            ].values,
            dtype=torch.long,
            device=device
        )

        edge_snapshot = (
            edges_df[
                edges_df["time"] == time_s
            ]
        )

        edge_index = build_edge_index(
            edge_snapshot,
            node_id_to_index,
            device
        )

        snapshots[time_s] = (
            x,
            edge_index,
            y,
            node_ids
        )

    return snapshots


# ============================================================
# STATIC TRAINING
# ============================================================

def train_static_model(
    model,
    snapshots,
    train_times,
    epochs=50,
    lr=0.01
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for time_s in train_times:

            x, edge_index, y, _ = (
                snapshots[time_s]
            )

            out = model(
                x,
                edge_index
            )

            loss = criterion(
                out,
                y
            )

            total_loss += loss

        total_loss = (
            total_loss /
            len(train_times)
        )

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

    return model


# ============================================================
# EVOLVEGCN-H TRAINING
#
# IMPORTANT:
# The temporal snapshots are processed sequentially.
#
# The loss is accumulated across the entire training
# sequence and the optimizer is updated once per epoch.
# This follows the temporal training procedure specified
# in the project guide.
# ============================================================

def train_evolvegcn(
    model,
    snapshots,
    train_times,
    device,
    epochs=50,
    lr=0.01
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    for epoch in range(epochs):

        model.train()

        # Reset temporal state at the beginning
        # of every training epoch.
        hidden = None

        total_loss = 0.0

        # Process snapshots in chronological order.
        for time_s in train_times:

            x, edge_index, y, _ = (
                snapshots[time_s]
            )

            out, hidden = model(
                x,
                edge_index,
                hidden
            )

            loss = criterion(
                out,
                y
            )

            total_loss = (
                total_loss +
                loss
            )

        # Average loss over all training snapshots.
        total_loss = (
            total_loss /
            len(train_times)
        )

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        if (
            (epoch + 1) % 10 == 0
            or epoch == 0
        ):

            print(
                f"EvolveGCN-H "
                f"Epoch {epoch + 1}/{epochs} "
                f"Loss: "
                f"{total_loss.item():.4f}"
            )

    return model


# ============================================================
# STATIC EVALUATION
# ============================================================

def evaluate_static_model(
    model,
    snapshots,
    test_times
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_nodes = []
    all_times = []

    with torch.no_grad():

        for time_s in test_times:

            x, edge_index, y, node_ids = (
                snapshots[time_s]
            )

            out = model(
                x,
                edge_index
            )

            predictions = (
                out.argmax(
                    dim=1
                )
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_nodes.extend(
                node_ids
            )

            all_times.extend(
                [time_s] *
                len(node_ids)
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_nodes),
        np.array(all_times)
    )


# ============================================================
# EVOLVEGCN-H EVALUATION
#
# The temporal state is warmed up using the complete training
# sequence before predictions are made on the test sequence.
# ============================================================

def evaluate_evolvegcn(
    model,
    snapshots,
    train_times,
    test_times
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_nodes = []
    all_times = []

    hidden = None

    with torch.no_grad():

        # ----------------------------------------------------
        # Warm up temporal state using training snapshots.
        # ----------------------------------------------------

        for time_s in train_times:

            x, edge_index, _, _ = (
                snapshots[time_s]
            )

            _, hidden = model(
                x,
                edge_index,
                hidden
            )

        # ----------------------------------------------------
        # Evaluate chronological test snapshots.
        # ----------------------------------------------------

        for time_s in test_times:

            x, edge_index, y, node_ids = (
                snapshots[time_s]
            )

            out, hidden = model(
                x,
                edge_index,
                hidden
            )

            predictions = (
                out.argmax(
                    dim=1
                )
            )

            all_labels.extend(
                y.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_nodes.extend(
                node_ids
            )

            all_times.extend(
                [time_s] *
                len(node_ids)
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_nodes),
        np.array(all_times)
    )


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(
    model_name,
    device
):

    if model_name == "Static-GCN":

        model = StaticGCN()

    elif model_name == "GraphSAGE":

        model = GraphSAGE()

    elif model_name == "GAT":

        model = GAT()

    elif model_name == "EvolveGCN-H":

        model = EvolveGCNH()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(device)


# ============================================================
# TEMPORAL ADVANTAGE
# ============================================================

def calculate_temporal_advantage(
    static_results,
    evolve_results,
    run
):

    static_labels = (
        static_results[0]
    )

    static_predictions = (
        static_results[1]
    )

    static_nodes = (
        static_results[2]
    )

    static_times = (
        static_results[3]
    )

    evolve_labels = (
        evolve_results[0]
    )

    evolve_predictions = (
        evolve_results[1]
    )

    evolve_nodes = (
        evolve_results[2]
    )

    evolve_times = (
        evolve_results[3]
    )

    static_df = pd.DataFrame(
        {
            "time": static_times,
            "node_id": static_nodes,
            "label": static_labels,
            "static_prediction":
                static_predictions,
        }
    )

    evolve_df = pd.DataFrame(
        {
            "time": evolve_times,
            "node_id": evolve_nodes,
            "label": evolve_labels,
            "evolve_prediction":
                evolve_predictions,
        }
    )

    merged = pd.merge(
        static_df,
        evolve_df,
        on=[
            "time",
            "node_id"
        ],
        suffixes=(
            "_static",
            "_evolve"
        )
    )

    # Same malicious node-time cases where
    # Static-GCN misses but EvolveGCN-H catches.
    same_node_time_caught = (
        (
            merged["label_static"] == 1
        )
        &
        (
            merged["static_prediction"] == 0
        )
        &
        (
            merged["evolve_prediction"] == 1
        )
    )

    static_missed = (
        (
            merged["label_static"] == 1
        )
        &
        (
            merged["static_prediction"] == 0
        )
    )

    evolve_caught = (
        (
            merged["label_evolve"] == 1
        )
        &
        (
            merged["evolve_prediction"] == 1
        )
    )

    return {
        "run":
            run,

        "static_gcn_missed_malicious":
            int(
                static_missed.sum()
            ),

        "evolvegcn_caught_malicious":
            int(
                evolve_caught.sum()
            ),

        "same_node_time_caught_by_evolvegcn":
            int(
                same_node_time_caught.sum()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(
            DEFAULT_DATA_DIR
        )
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(
            DEFAULT_RESULTS_DIR
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    results_dir = Path(
        args.results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    elif torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    nodes_df, edges_df = load_data(
        data_dir,
        device
    )

    runs = sorted(
        nodes_df["run"]
        .unique()
    )

    print(
        f"Runs: {runs}"
    )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    result_rows = []

    prediction_details = {}

    temporal_rows = []

    # --------------------------------------------------------
    # Process each run
    # --------------------------------------------------------

    for run in runs:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"RUN {run}"
        )

        print(
            "=" * 60
        )

        run_nodes = nodes_df[
            nodes_df["run"] == run
        ].copy()

        run_edges = edges_df[
            edges_df["run"] == run
        ].copy()

        all_times = sorted(
            run_nodes["time"]
            .unique()
        )

        train_times, test_times = (
            split_temporal(
                all_times
            )
        )

        print(
            f"Train snapshots: "
            f"{len(train_times)}"
        )

        print(
            f"Test snapshots: "
            f"{len(test_times)}"
        )

        # ----------------------------------------------------
        # Fit scaler ONLY on training snapshots.
        # ----------------------------------------------------

        scaler = fit_scaler(
            run_nodes,
            train_times
        )

        # ----------------------------------------------------
        # Prepare all snapshots using train-fitted scaler.
        # ----------------------------------------------------

        snapshots = prepare_snapshots(
            run_nodes,
            run_edges,
            all_times,
            scaler,
            device
        )

        # ----------------------------------------------------
        # Store model outputs for temporal comparison.
        # ----------------------------------------------------

        static_result_for_temporal = None
        evolve_result_for_temporal = None

        # ----------------------------------------------------
        # Train/evaluate each model.
        # ----------------------------------------------------

        for model_name in MODEL_NAMES:

            print(
                "\n"
                + "-" * 50
            )

            print(
                f"Model: {model_name}"
            )

            print(
                "-" * 50
            )

            model = create_model(
                model_name,
                device
            )

            # ------------------------------------------------
            # Parameter count
            # ------------------------------------------------

            parameter_count = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            print(
                f"Trainable parameters: "
                f"{parameter_count}"
            )

            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            if model_name == "EvolveGCN-H":

                model = train_evolvegcn(
                    model,
                    snapshots,
                    train_times,
                    device,
                    epochs=args.epochs
                )

            else:

                model = train_static_model(
                    model,
                    snapshots,
                    train_times,
                    epochs=args.epochs
                )

            # ------------------------------------------------
            # Evaluation
            # ------------------------------------------------

            if model_name == "EvolveGCN-H":

                evaluation = (
                    evaluate_evolvegcn(
                        model,
                        snapshots,
                        train_times,
                        test_times
                    )
                )

                evolve_result_for_temporal = (
                    evaluation
                )

            else:

                evaluation = (
                    evaluate_static_model(
                        model,
                        snapshots,
                        test_times
                    )
                )

                if model_name == "Static-GCN":

                    static_result_for_temporal = (
                        evaluation
                    )

            labels = evaluation[0]

            predictions = evaluation[1]

            metrics = calculate_metrics(
                labels,
                predictions
            )

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"Malicious Precision: "
                f"{metrics['mal_precision']:.4f}"
            )

            print(
                f"Malicious Recall: "
                f"{metrics['mal_recall']:.4f}"
            )

            print(
                f"Malicious F1: "
                f"{metrics['mal_f1']:.4f}"
            )

            # ------------------------------------------------
            # Save aggregate result.
            # ------------------------------------------------

            result_rows.append(
                {
                    "run":
                        run,

                    "model":
                        model_name,

                    "accuracy":
                        metrics["accuracy"],

                    "mal_precision":
                        metrics["mal_precision"],

                    "mal_recall":
                        metrics["mal_recall"],

                    "mal_f1":
                        metrics["mal_f1"],
                }
            )

            # ------------------------------------------------
            # Save prediction details.
            # ------------------------------------------------

            detail_df = pd.DataFrame(
                {
                    "run":
                        run,

                    "model":
                        model_name,

                    "time":
                        evaluation[3],

                    "node_id":
                        evaluation[2],

                    "label":
                        evaluation[0],

                    "prediction":
                        evaluation[1],
                }
            )

            if run not in prediction_details:

                prediction_details[run] = []

            prediction_details[
                run
            ].append(
                detail_df
            )

        # ----------------------------------------------------
        # Temporal advantage comparison.
        # ----------------------------------------------------

        if (
            static_result_for_temporal
            is not None
            and
            evolve_result_for_temporal
            is not None
        ):

            temporal_result = (
                calculate_temporal_advantage(
                    static_result_for_temporal,
                    evolve_result_for_temporal,
                    run
                )
            )

            temporal_rows.append(
                temporal_result
            )

            print(
                "\nTemporal advantage:"
            )

            print(
                temporal_result
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        result_rows
    )

    results_path = (
        results_dir /
        "gnn_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(
        f"\nSaved: {results_path}"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby("model")
        [
            [
                "accuracy",
                "mal_precision",
                "mal_recall",
                "mal_f1",
            ]
        ]
        .agg(
            [
                "mean",
                "std"
            ]
        )
    )

    summary_path = (
        results_dir /
        "gnn_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    print(
        f"Saved: {summary_path}"
    )

    # ========================================================
    # PREDICTION DETAIL FILES
    # ========================================================

    for run, frames in (
        prediction_details.items()
    ):

        combined = pd.concat(
            frames,
            ignore_index=True
        )

        path = (
            results_dir /
            f"prediction_detail_run{run}.csv"
        )

        combined.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

    # ========================================================
    # TEMPORAL ADVANTAGE CSV
    # ========================================================

    temporal_df = pd.DataFrame(
        temporal_rows
    )

    temporal_path = (
        results_dir /
        "temporal_advantage.csv"
    )

    temporal_df.to_csv(
        temporal_path,
        index=False
    )

    print(
        f"Saved: {temporal_path}"
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "GNN TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

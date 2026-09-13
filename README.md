# MANET-FL-GNN

Dynamic intrusion detection for Mobile Ad-hoc Networks (MANETs) using Graph Neural Networks (GNNs) and Federated Learning (FL).

This project builds a full pipeline — from raw NS-3 mobility simulation to a time-evolving graph dataset, classical ML baselines, several GNN architectures, a federated-learning variant, and an interactive dashboard — to detect black-hole, grey-hole, and wormhole routing attacks in a MANET as the network topology changes over time.


---

## Table of Contents

- [Overview](#overview)
- [Project Pipeline](#project-pipeline)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
  - [Mobility Simulation (NS-3)](#mobility-simulation-ns-3)
  - [Dynamic Graph Construction](#dynamic-graph-construction)
  - [Attack Model](#attack-model)
  - [Node Features and Labels](#node-features-and-labels)
- [Models](#models)
  - [Classical Baselines](#classical-baselines)
  - [Graph Neural Networks](#graph-neural-networks)
  - [Federated GCN](#federated-gcn)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate / Inspect Mobility Data](#1-generate--inspect-mobility-data)
  - [2. Build the Dynamic Graph Dataset](#2-build-the-dynamic-graph-dataset)
  - [3. Validate the Dataset](#3-validate-the-dataset)
  - [4. Train Classical Baselines](#4-train-classical-baselines)
  - [5. Train GNN Models](#5-train-gnn-models)
  - [6. Run Scalability Benchmarks](#6-run-scalability-benchmarks)
  - [7. Federated GCN Notebook](#7-federated-gcn-notebook)
  - [8. Visualize Topology Snapshots](#8-visualize-topology-snapshots)
  - [9. Interactive Dashboard](#9-interactive-dashboard)
- [Notes and Known Limitations](#notes-and-known-limitations)
- [License](#license)

---

## Overview

MANETs are wireless networks with no fixed infrastructure — nodes move freely and forward each other's traffic. This mobility makes them vulnerable to routing attacks (black-hole, grey-hole, wormhole) that are hard to detect with static, snapshot-based classifiers because the topology, node roles, and attack behaviour all change over time.

This project treats intrusion detection as a **dynamic node-classification problem on an evolving graph**:

1. Simulate realistic node mobility with **NS-3** (Random Waypoint model).
2. Turn raw position traces into a **time-series of communication graphs**, injecting time-triggered black-hole / grey-hole / wormhole attacks and simulated multi-hop packet forwarding.
3. Derive per-node, per-timestep traffic features (drop rate, forwarding ratio, trust, energy, etc.).
4. Compare classical tabular ML baselines against static and temporal **Graph Neural Networks** (GCN, GraphSAGE, GAT, and a custom EvolveGCN-H-style model) on a **temporal train/test split** (train on early snapshots, test on later ones — the attacks only "turn on" partway through the simulation).
5. Prototype a **federated learning** version of the GCN, where each simulation run acts as an independent client and only model weights are averaged across clients (FedAvg).
6. Benchmark **scalability** from 100 to 500 nodes.
7. Explore results interactively in a **Streamlit dashboard**.

## Project Pipeline

```
NS-3 mobility sim  ─────►  Raw position CSVs  ─────►  Dynamic graph builder  ─────►  nodes_dynamic.csv
(manet_mobility.cc)        (data/raw/)                (build_dynamic_dataset.py)     edges_dynamic.csv
                                                                                       (data/processed/)
                                                                                             │
                     ┌───────────────────────────────────────────────────────────────────────┤
                     ▼                                   ▼                                   ▼
          Classical baselines                    GNN models                        Federated GCN
      (RandomForest / SVM / XGBoost)   (Static-GCN / GraphSAGE / GAT /         (per-run clients + FedAvg,
        train_baselines.py               EvolveGCN-H fallback)                  federated_gnn notebook)
                     │                   train_gnn_models.py                                │
                     └───────────────────────────────┬────────────────────────────────────┘
                                                       ▼
                                          results/*.csv, *.png
                                                       │
                                                       ▼
                                       Streamlit dashboard (dashboard/streamlit_app.py)
```

## Repository Structure

```
manet-fl-gnn-review/
├── ns3/
│   ├── manet_mobility.cc          # NS-3 scratch source for the mobility simulation
│   └── README_ADRITO.md           # Notes on the NS-3 component (config, validation, repro commands)
├── manet_mobility.cc              # Duplicate of the NS-3 source at repo root
├── python/
│   ├── build_dynamic_dataset.py   # Raw positions -> dynamic graph (nodes_dynamic.csv, edges_dynamic.csv)
│   ├── check_dataset.py           # Sanity-checks the processed dataset and temporal split
│   ├── train_baselines.py         # RandomForest / SVM / XGBoost baselines on node features
│   ├── train_gnn_models.py        # Static-GCN, GraphSAGE, GAT, EvolveGCN-H training + evaluation
│   ├── scalability.py             # GCN training-time / accuracy benchmark for 100–500 node graphs
│   └── plot_snapshots.py          # Renders topology snapshots (nodes + edges) at chosen timestamps
├── federated_gnn/
│   └── FL_GCN_MANET.ipynb         # Federated learning (FedAvg) GCN experiment, Colab-oriented
├── dashboard/
│   ├── streamlit_app.py           # Interactive dashboard over processed data + saved results
│   └── requirements.txt           # Dashboard-only dependencies
├── data/
│   ├── raw/                       # NS-3 position CSVs (baseline + scalability sweeps)
│   │   └── scalability/
│   └── processed/                 # Dynamic dataset + per-size graphs used for scalability
│       ├── nodes_dynamic.csv
│       ├── edges_dynamic.csv
│       └── {100,200,300,400,500}/{nodes.csv,edges.csv}
└── results/                        # All metrics, summaries, and topology plots produced by the scripts
    ├── baseline_results.csv / baseline_summary.csv
    ├── gnn_results.csv / gnn_summary.csv
    ├── fl_gcn_5runs.csv / fl_gcn_summary.csv
    ├── temporal_advantage.csv
    ├── scalability.csv
    ├── prediction_detail_run{1..5}.csv
    ├── scalability_accuracy.png / scalability_time.png
    └── topology_run1_t{0,150,250,350,450}.png
```

## Dataset

### Mobility Simulation (NS-3)

Node mobility is generated in NS-3 (`ns3/manet_mobility.cc`) using the **Random Waypoint** mobility model:

| Parameter | Value |
|---|---|
| NS-3 version | 3.47 |
| Simulation area | 1000 × 1000 m |
| Node speed | uniform 5–25 m/s |
| Pause time | 10 s |
| Simulation duration | 500 s |
| Snapshot interval | 10 s (51 snapshots: t = 0, 10, …, 500) |

**Baseline dataset:** 150 nodes × 5 independent runs (RNG seed 12345, run numbers 1–5) → 7,650 rows per run, 38,250 rows total.

**Scalability dataset:** single runs at 100 / 200 / 300 / 400 / 500 nodes with the same mobility configuration, used only for the scalability benchmark.

Each raw position CSV (`data/raw/*.csv`) has the schema:

```
run,time,node_id,x,y,speed
1,0.00,0,389.63,402.33,0.00
1,0.00,1,296.83,120.89,0.00
```

See `ns3/README_ADRITO.md` for the exact reproduction commands and the validation checks applied to the raw data (row counts, coordinate bounds, non-negative speeds, etc.).

### Dynamic Graph Construction

`python/build_dynamic_dataset.py` turns the raw position traces into a **time-series of communication graphs** and simulates traffic on top of them:

- **Edges** are formed between any two nodes within a configurable radio **communication range** (default 250 m) at each snapshot.
- **Traffic** is simulated as a configurable number of random source→destination **flows** per snapshot (default 120 flows × 20 packets), forwarded hop-by-hop along the shortest path (BFS) between source and destination.
- **Packet drops** at each hop depend on whether the forwarding node is currently acting maliciously (see [Attack Model](#attack-model)).
- Node-level statistics (packets sent/received/forwarded/dropped, path hits, per-hop delay, degree, cumulative energy consumption, a derived **trust** score) are aggregated per node, per snapshot, per run and written to `data/processed/nodes_dynamic.csv`.
- All graph edges per snapshot (including synthetic wormhole shortcut edges) are written to `data/processed/edges_dynamic.csv`.

### Attack Model

Attacks are **time-triggered** — a subset of nodes is fixed for the whole run, but each attack type only activates after a threshold time, so the same physical topology looks "normal" early on and "attacked" later:

| Time | Attack activates | Behaviour |
|---|---|---|
| t ≥ 100 s | Black-hole (~4% of nodes) | drops ~95% of forwarded packets |
| t ≥ 200 s | Grey-hole (~4% of nodes) | drops ~50% of forwarded packets |
| t ≥ 300 s | Wormhole (1–2 colluding pairs) | adds an out-of-band shortcut edge between the pair, drops ~15% of packets, and injects artificially low hop delay |

This design is what makes the **temporal train/test split** meaningful: models trained only on the early (mostly attack-free) snapshots must generalize to detect attacks they never saw activate during training.

### Node Features and Labels

Each row of `nodes_dynamic.csv` is one node at one snapshot, with columns:

| Column | Description |
|---|---|
| `run`, `time`, `node_id` | Identifiers |
| `energy` | Simulated remaining battery, decreasing with cumulative forwarding/receiving work |
| `trust` | Derived score combining drop rate and routing load, with attack-specific penalties |
| `drop_rate` | Fraction of packets received but not forwarded by this node |
| `mobility` | Instantaneous node speed (m/s) |
| `forward_ratio` | Fraction of received packets successfully forwarded |
| `delay` | Average simulated per-hop delay |
| `degree` | Number of radio-range neighbours |
| `packets_received` / `packets_forwarded` / `packets_dropped` | Raw traffic counters |
| `route_hits` | Number of times this node appeared on a forwarding path |
| `label` | **0 = normal, 1 = malicious** (any active attack) |
| `attack` | Attack type string: `normal`, `blackhole`, `greyhole`, `wormhole` |

The four features used by every model (baselines and GNNs alike) are: **`drop_rate`, `trust`, `forward_ratio`, `energy`**.

## Models

All models are evaluated with a **per-run temporal split**: for each simulation run, the first 70% of timestamps are used for training and the last 30% for testing, so the test period always includes malicious behaviour that never appeared during training.

### Classical Baselines

`python/train_baselines.py` trains, per run, three tabular classifiers on the 4-feature node table:

- **Random Forest** (250 trees, balanced class weights)
- **SVM** (RBF kernel, standardized features, balanced class weights)
- **XGBoost** (250 trees, depth 4, learning rate 0.05)

### Graph Neural Networks

`python/train_gnn_models.py` builds one graph per (run, timestamp) using `edges_dynamic.csv` for connectivity and the 4 scaled node features as node attributes, then trains/evaluates four models per run:

- **Static-GCN** — 2-layer `GCNConv` (4 → 16 → 2), trained per-snapshot with no memory across time.
- **GraphSAGE** — 2-layer `SAGEConv`, same shape.
- **GAT** — 2-layer `GATConv` (single attention head), same shape.
- **EvolveGCN-H (custom fallback)** — a from-scratch, EvolveGCN-H-*style* model that evolves its first-layer GCN weight matrix over time via a GRU cell driven by pooled node embeddings. **Note:** the official `torch_geometric_temporal` implementation could not be installed in the project environment (a `torch-sparse` build failure), so this is a custom re-implementation of the idea, not the reference EvolveGCN-H class.

The script also computes a **temporal advantage** comparison (`results/temporal_advantage.csv`): how many malicious node-snapshots the static GCN misses that the evolving model catches.

### Federated GCN

`federated_gnn/FL_GCN_MANET.ipynb` (designed to run in Google Colab) implements **FedAvg** over the same lightweight 2-layer GCN architecture:

- Each of the 5 simulation runs acts as an independent **client** with its own local training graphs.
- Each round, every client trains locally from the current global weights, and the server averages client weights (`federated_average`) to produce the next global model.
- A class-imbalance weight for the malicious class is swept (2.0 / 4.0 / 6.0 / 8.0) and the best value is used for the final 5-round, 5-seed experiment, whose results are saved to `results/fl_gcn_5runs.csv` and `results/fl_gcn_summary.csv`.

## Results

Mean over 5 runs, malicious-class precision/recall/F1 (from `results/*_summary.csv`):

| Model | Accuracy | Malicious Precision | Malicious Recall | Malicious F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.968 | 0.895 | 0.793 | 0.841 |
| SVM | 0.961 | 0.818 | 0.838 | 0.822 |
| XGBoost | 0.977 | 0.998 | 0.788 | 0.881 |
| Static-GCN | 0.893 | 0.000 | 0.000 | 0.000 |
| GAT | 0.890 | 0.233 | 0.009 | 0.016 |
| GraphSAGE | **0.972** | **0.999** | **0.743** | **0.851** |
| EvolveGCN-H (custom) | 0.893 | 0.527 | 0.020 | 0.038 |
| Federated GCN (FedAvg) | 0.704 | 0.189 | 0.534 | 0.278 |

Key observations from these numbers:

- The tree-based baselines (XGBoost, Random Forest) and **GraphSAGE** are the strongest detectors — GraphSAGE is the only GNN that comes close to matching the tabular baselines, likely because its neighbourhood-sampling aggregation is less sensitive to the highly dynamic graph structure than GCN/GAT's spectral-style convolutions.
- **Static-GCN and GAT essentially collapse to predicting "normal"** on the temporal test split (near-zero recall), showing that a snapshot-only GCN struggles when attacks are entirely absent from the training window.
- The custom **EvolveGCN-H** model catches a small number of malicious node-snapshots the static GCN completely misses (`results/temporal_advantage.csv` — up to 13 out of 256 missed cases per run), a modest but real temporal advantage, though its overall recall is still low.
- The **federated GCN** trades a lot of accuracy for higher malicious recall than every centralized GNN except GraphSAGE, but with low precision — consistent with each client seeing a different, non-IID mix of attack types and averaging pulling the global model toward a more attack-sensitive but noisier decision boundary.
- **Scalability** (`results/scalability.csv`): the 114-parameter static GCN trains in well under a second even at 500 nodes / ~39k edges, confirming the architecture itself is not the bottleneck — the harder problem is detection quality under temporal drift, not raw compute.

Plots are saved under `results/`: `scalability_accuracy.png`, `scalability_time.png`, and topology snapshots `topology_run1_t{0,150,250,350,450}.png` showing the network at each attack-activation milestone.

## Installation

```bash
git clone https://github.com/Raghavfw/MANET.git
cd MANET
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas scikit-learn xgboost matplotlib torch torch_geometric
pip install -r dashboard/requirements.txt   # streamlit, plotly, networkx, etc.
```

Notes:

- `torch_geometric` requires a matching PyTorch build; follow the [official install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your OS/CUDA version.
- The NS-3 mobility simulation itself requires a working **NS-3.47** install (see `ns3/README_ADRITO.md`); this is only needed if you want to regenerate raw position data — the repo already ships pre-generated CSVs under `data/raw/` and `data/processed/`.
- `torch_geometric_temporal` is **not** required — the EvolveGCN-H model in this repo is a custom fallback (see [Graph Neural Networks](#graph-neural-networks)).

## Usage

All commands below are run from the repository root and assume the pre-generated data under `data/` is present.

### 1. Generate / Inspect Mobility Data

To regenerate raw mobility traces from scratch (requires NS-3.47 installed separately):

```bash
./ns3 run "scratch/manet_mobility --nodes=150 --simTime=500 --interval=10 \
  --minSpeed=5 --maxSpeed=25 --pause=10 --areaX=1000 --areaY=1000 \
  --run=1 --output=positions_150n_seed1.csv"
```

Repeat with `--run=2..5` for the remaining baseline seeds, and with `--nodes=100/200/300/400/500` for the scalability sweep. Otherwise, skip this step — `data/raw/` already contains the generated CSVs.

### 2. Build the Dynamic Graph Dataset

```bash
python python/build_dynamic_dataset.py \
  --positions data/raw/positions_150n_seed1.csv data/raw/positions_150n_seed2.csv \
              data/raw/positions_150n_seed3.csv data/raw/positions_150n_seed4.csv \
              data/raw/positions_150n_seed5.csv \
  --out-nodes data/processed/nodes_dynamic.csv \
  --out-edges data/processed/edges_dynamic.csv \
  --range 250 --flows 120 --packets 20 --seed 2026
```

Key flags: `--range` (radio range in metres), `--flows` (random flows simulated per snapshot), `--packets` (packets per flow), `--seed` (base RNG seed, offset per run internally).

### 3. Validate the Dataset

```bash
python python/check_dataset.py --nodes data/processed/nodes_dynamic.csv
```

Confirms column schema, per-run node/time coverage, attack-label distribution, and that every run's temporal test split contains both classes (exits non-zero otherwise).

### 4. Train Classical Baselines

```bash
python python/train_baselines.py \
  --nodes data/processed/nodes_dynamic.csv \
  --out results/baseline_results.csv
```

Writes per-run, per-model metrics to `results/baseline_results.csv` and the mean/std summary to `results/baseline_summary.csv`.

### 5. Train GNN Models

```bash
python python/train_gnn_models.py \
  --data-dir data/processed \
  --results-dir results \
  --epochs 50
```

Trains Static-GCN, GraphSAGE, GAT, and the custom EvolveGCN-H per run, and writes:
- `results/gnn_results.csv` / `results/gnn_summary.csv` — per-run and aggregated metrics
- `results/prediction_detail_run{N}.csv` — every node/timestamp prediction, for error analysis
- `results/temporal_advantage.csv` — static-GCN-vs-EvolveGCN-H missed/caught comparison

Runs on CUDA or Apple MPS automatically if available, otherwise CPU.

### 6. Run Scalability Benchmarks

```bash
python python/scalability.py
```

Trains the same 114-parameter 2-layer GCN on the pre-built `data/processed/{100,200,300,400,500}/{nodes,edges}.csv` graphs for 10 epochs each, recording accuracy and wall-clock training time to `results/scalability.csv` (and the corresponding plots).

### 7. Federated GCN Notebook

Open `federated_gnn/FL_GCN_MANET.ipynb` in Jupyter or Google Colab. The notebook expects `nodes_dynamic.csv` and `edges_dynamic.csv` to be uploaded (it includes a Colab file-upload cell); if running locally, replace that cell with a direct `pd.read_csv(...)` pointing at `data/processed/`. It walks through: temporal split → per-client graph construction → local training + FedAvg → class-weight sweep → final 5-seed evaluation → saving `results/fl_gcn_5runs.csv` and `results/fl_gcn_summary.csv`.

### 8. Visualize Topology Snapshots

```bash
python python/plot_snapshots.py \
  --positions data/raw/positions_150n_seed1.csv \
  --nodes data/processed/nodes_dynamic.csv \
  --edges data/processed/edges_dynamic.csv \
  --run 1 --times 0 150 250 350 450 \
  --out results
```

Renders one PNG per requested timestamp (default: right before each attack activates), colouring nodes by their current label/attack type — used to produce `results/topology_run1_t{0,150,250,350,450}.png`.

### 9. Interactive Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/streamlit_app.py
```

Run this from the repository root (the app reads `data/processed/` and `results/` with relative paths). The dashboard lets you pick a simulation run and timestamp, inspect the live topology and node states, and browse the saved GNN/FL summary metrics.

## Notes and Known Limitations

- **EvolveGCN-H is a custom implementation**, not the reference `torch_geometric_temporal` model — see [Graph Neural Networks](#graph-neural-networks) for why, and treat its results as indicative of a temporal-weight-evolution approach rather than the published architecture's exact performance.
- The attack model, traffic simulation, and trust/energy features are **synthetic approximations** built on top of real NS-3 mobility traces, not a full network-stack simulation (e.g. there is no MAC-layer contention, real routing protocol, or interference model) — treat absolute performance numbers as a controlled benchmark for comparing detection methods, not as production intrusion-detection accuracy.
- The federated learning setup uses only 5 clients (one per simulation run) with plain FedAvg and no differential privacy, secure aggregation, or client dropout simulation.
- `data/raw/*.csv:Zone.Identifier` files present in some environments are Windows metadata artifacts from downloading the CSVs and can be safely ignored/deleted.


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="MANET Intrusion Detection",
    layout="wide"
)

st.title("Dynamic MANET Intrusion Detection Demo")

st.write(
    "Interactive visualization of the dynamic MANET dataset and saved model results."
)

@st.cache_data
def load_data():
    nodes = pd.read_csv("data/processed/nodes_dynamic.csv")
    edges = pd.read_csv("data/processed/edges_dynamic.csv")
    fl = pd.read_csv("results/fl_gcn_summary.csv")

    gnn = pd.read_csv(
        "results/gnn_summary.csv",
        header=[0, 1],
        index_col=0
    )

    gnn.columns = [
        "Accuracy Mean",
        "Accuracy Std",
        "Malicious Precision Mean",
        "Malicious Precision Std",
        "Malicious Recall Mean",
        "Malicious Recall Std",
        "Malicious F1 Mean",
        "Malicious F1 Std"
    ]

    gnn = gnn.reset_index().rename(
        columns={"model": "Model"}
    )

    return nodes, edges, fl, gnn

nodes, edges, fl_summary, gnn_summary = load_data()

st.sidebar.header("Simulation Controls")

runs = sorted(nodes["run"].unique())

selected_run = st.sidebar.selectbox(
    "Select simulation run",
    runs
)

run_nodes = nodes[
    nodes["run"] == selected_run
]

times = sorted(run_nodes["time"].unique())

selected_time = st.sidebar.select_slider(
    "Select time (seconds)",
    options=times
)

snapshot_nodes = nodes[
    (nodes["run"] == selected_run) &
    (nodes["time"] == selected_time)
].copy()

snapshot_edges = edges[
    (edges["run"] == selected_run) &
    (edges["time"] == selected_time)
].copy()

normal_count = (
    snapshot_nodes["label"] == 0
).sum()

malicious_count = (
    snapshot_nodes["label"] == 1
).sum()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Time",
    f"{selected_time} s"
)

col2.metric(
    "Normal Nodes",
    int(normal_count)
)

col3.metric(
    "Malicious Nodes",
    int(malicious_count)
)

x_col = None
y_col = None

for name in ["x", "x_pos", "pos_x", "x_position"]:
    if name in snapshot_nodes.columns:
        x_col = name
        break

for name in ["y", "y_pos", "pos_y", "y_position"]:
    if name in snapshot_nodes.columns:
        y_col = name
        break

if x_col is None or y_col is None:
    snapshot_nodes["plot_x"] = snapshot_nodes["node_id"] % 15
    snapshot_nodes["plot_y"] = snapshot_nodes["node_id"] // 15
    x_col = "plot_x"
    y_col = "plot_y"

fig = go.Figure()

positions = {}

for _, row in snapshot_nodes.iterrows():
    positions[int(row["node_id"])] = (
        row[x_col],
        row[y_col]
    )

for _, edge in snapshot_edges.iterrows():
    src = int(edge["src"])
    dst = int(edge["dst"])

    if src in positions and dst in positions:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=0.5,
                    color="rgba(150,150,150,0.25)"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )

normal = snapshot_nodes[
    snapshot_nodes["label"] == 0
]

fig.add_trace(
    go.Scatter(
        x=normal[x_col],
        y=normal[y_col],
        mode="markers",
        name="Normal",
        marker=dict(
            size=9
        ),
        text=[
            f"Node {node_id}"
            for node_id in normal["node_id"]
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

malicious = snapshot_nodes[
    snapshot_nodes["label"] == 1
]

fig.add_trace(
    go.Scatter(
        x=malicious[x_col],
        y=malicious[y_col],
        mode="markers",
        name="Malicious",
        marker=dict(
            size=12,
            symbol="x"
        ),
        text=[
            f"Node {node_id}<br>Attack: {attack}"
            for node_id, attack in zip(
                malicious["node_id"],
                malicious["attack"]
            )
        ],
        hovertemplate="%{text}<extra></extra>"
    )
)

fig.update_layout(
    title=(
        f"Dynamic MANET — Run {selected_run}, "
        f"Time {selected_time}s"
    ),
    height=650,
    xaxis_title="X position",
    yaxis_title="Y position",
    showlegend=True
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Attack Distribution")

if malicious_count == 0:
    st.info(
        "No malicious nodes at this timestamp."
    )
else:
    attack_counts = (
        malicious["attack"]
        .value_counts()
        .reset_index()
    )

    attack_counts.columns = [
        "Attack",
        "Nodes"
    ]

    st.dataframe(
        attack_counts,
        use_container_width=True
    )

st.subheader("Federated Lightweight GCN Results")

st.dataframe(
    fl_summary,
    use_container_width=True
)

st.subheader("Centralized GNN Model Results")

st.dataframe(
    gnn_summary,
    use_container_width=True
)

st.caption(
    "This dashboard visualizes saved simulation data and model results."
)

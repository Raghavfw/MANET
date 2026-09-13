import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ap = argparse.ArgumentParser()

ap.add_argument(
    "--positions",
    required=True
)

ap.add_argument(
    "--nodes",
    default="data/processed/nodes_dynamic.csv"
)

ap.add_argument(
    "--edges",
    default="data/processed/edges_dynamic.csv"
)

ap.add_argument(
    "--run",
    type=int,
    default=1
)

ap.add_argument(
    "--times",
    nargs="+",
    type=float,
    default=[0, 150, 250, 350, 450]
)

ap.add_argument(
    "--out",
    default="results"
)

args = ap.parse_args()

pos = pd.read_csv(
    args.positions
)

nodes = pd.read_csv(
    args.nodes
)

edges = pd.read_csv(
    args.edges
)

Path(
    args.out
).mkdir(
    parents=True,
    exist_ok=True
)

for t in args.times:

    p = pos[
        (pos.run == args.run)
        & (pos.time == t)
    ].sort_values(
        "node_id"
    )

    n = nodes[
        (nodes.run == args.run)
        & (nodes.time == t)
    ].sort_values(
        "node_id"
    )

    e = edges[
        (edges.run == args.run)
        & (edges.time == t)
    ]

    if p.empty:

        print(
            f"Skipping t={t}: "
            "snapshot not found"
        )

        continue

    xy = p.set_index(
        "node_id"
    )[["x", "y"]]

    plt.figure(
        figsize=(8, 7)
    )

    for row in e.itertuples():

        a = xy.loc[row.src]
        b = xy.loc[row.dst]

        plt.plot(
            [a.x, b.x],
            [a.y, b.y],
            linewidth=0.35,
            alpha=0.25
        )

    normal = n[
        n.label == 0
    ].node_id

    bad = n[
        n.label == 1
    ].node_id

    plt.scatter(
        xy.loc[normal].x,
        xy.loc[normal].y,
        s=16,
        label="Normal"
    )

    if len(bad):

        plt.scatter(
            xy.loc[bad].x,
            xy.loc[bad].y,
            s=28,
            marker="x",
            label="Malicious"
        )

    plt.title(
        f"Dynamic MANET snapshot - "
        f"run {args.run}, t={t:.0f}s"
    )

    plt.xlabel(
        "X position (m)"
    )

    plt.ylabel(
        "Y position (m)"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        Path(args.out)
        / f"topology_run{args.run}_t{int(t)}.png"
    )

    plt.savefig(
        path,
        dpi=180
    )

    plt.close()

    print(
        "Saved",
        path
    )


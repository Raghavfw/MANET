"""Build a dynamic MANET intrusion dataset from ns-3 position snapshots.

The mobility/topology comes from ns-3. This post-processing stage constructs
communication links using a radio range, runs simple multi-hop packet
forwarding, and introduces time-dependent attack behaviour:

100 s+: black-hole nodes
200 s+: grey-hole nodes
300 s+: wormhole pair(s)

Outputs:
nodes_dynamic.csv - per-node features and labels per time snapshot
edges_dynamic.csv - dynamic graph edges, including wormhole shortcut edges
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


def build_neighbors(points: np.ndarray, comm_range: float):
    n = len(points)
    adj = [[] for _ in range(n)]
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))

            if d <= comm_range:
                adj[i].append(j)
                adj[j].append(i)

                edges.append((i, j, d, "radio"))

    return adj, edges


def shortest_path(adj, src: int, dst: int):

    if src == dst:
        return [src]

    parent = [-1] * len(adj)
    parent[src] = src

    q = deque([src])

    while q:

        u = q.popleft()

        for v in adj[u]:

            if parent[v] != -1:
                continue

            parent[v] = u

            if v == dst:
                q.clear()
                break

            q.append(v)

    if parent[dst] == -1:
        return None

    path = [dst]

    while path[-1] != src:
        path.append(parent[path[-1]])

    path.reverse()

    return path


def choose_attack_nodes(n: int, rng: np.random.Generator):

    ids = np.arange(n)
    rng.shuffle(ids)

    # About 4% black-hole, 4% grey-hole,
    # and 2 wormhole pairs for large networks.
    k_black = max(2, int(round(0.04 * n)))
    k_grey = max(2, int(round(0.04 * n)))

    k_pairs = 2 if n >= 100 else 1

    need = k_black + k_grey + 2 * k_pairs

    if need >= n:
        raise ValueError(
            "Too few nodes for the requested attack configuration"
        )

    black = set(ids[:k_black].tolist())

    grey = set(
        ids[k_black:k_black + k_grey].tolist()
    )

    rest = ids[
        k_black + k_grey:
        k_black + k_grey + 2 * k_pairs
    ].tolist()

    worm_pairs = [
        (rest[i], rest[i + 1])
        for i in range(0, len(rest), 2)
    ]

    worm_nodes = {
        v
        for pair in worm_pairs
        for v in pair
    }

    return black, grey, worm_pairs, worm_nodes


def active_attack(
    node: int,
    time_s: float,
    black,
    grey,
    worm_nodes
):

    if time_s >= 100 and node in black:
        return "blackhole"

    if time_s >= 200 and node in grey:
        return "greyhole"

    if time_s >= 300 and node in worm_nodes:
        return "wormhole"

    return "normal"


def add_wormholes(
    adj,
    edges,
    points,
    worm_pairs,
    time_s
):

    if time_s < 300:
        return

    for a, b in worm_pairs:

        if b not in adj[a]:

            adj[a].append(b)
            adj[b].append(a)

            dist = float(
                np.linalg.norm(points[a] - points[b])
            )

            edges.append(
                (a, b, dist, "wormhole")
            )


def process_run(
    run_df: pd.DataFrame,
    comm_range: float,
    flows: int,
    packets_per_flow: int,
    seed: int
):

    run_id = int(run_df["run"].iloc[0])

    rng = np.random.default_rng(
        seed + run_id * 1000
    )

    n = int(run_df["node_id"].max()) + 1

    black, grey, worm_pairs, worm_nodes = \
        choose_attack_nodes(n, rng)

    base_energy = rng.uniform(
        0.85,
        1.0,
        size=n
    )

    cumulative_work = np.zeros(
        n,
        dtype=float
    )

    node_rows = []
    edge_rows = []

    for time_s in sorted(
        run_df["time"].unique()
    ):

        snap = (
            run_df[
                run_df["time"] == time_s
            ]
            .sort_values("node_id")
        )

        if len(snap) != n:
            raise ValueError(
                f"Run {run_id}, t={time_s}: "
                f"expected {n} nodes but found "
                f"{len(snap)}"
            )

        points = snap[
            ["x", "y"]
        ].to_numpy(
            dtype=float
        )

        speed = snap[
            "speed"
        ].to_numpy(
            dtype=float
        )

        adj, edges = build_neighbors(
            points,
            comm_range
        )

        add_wormholes(
            adj,
            edges,
            points,
            worm_pairs,
            float(time_s)
        )

        for u, v, dist, etype in edges:

            edge_rows.append([
                run_id,
                time_s,
                u,
                v,
                etype,
                dist
            ])

        received = np.zeros(
            n,
            dtype=int
        )

        forwarded = np.zeros(
            n,
            dtype=int
        )

        dropped = np.zeros(
            n,
            dtype=int
        )

        sent = np.zeros(
            n,
            dtype=int
        )

        path_hits = np.zeros(
            n,
            dtype=int
        )

        delay_sum = np.zeros(
            n,
            dtype=float
        )

        delay_count = np.zeros(
            n,
            dtype=int
        )

        # Random source-destination traffic
        # over the current topology.
        for _ in range(flows):

            src, dst = rng.choice(
                n,
                size=2,
                replace=False
            )

            path = shortest_path(
                adj,
                int(src),
                int(dst)
            )

            if not path or len(path) < 2:
                continue

            current = packets_per_flow

            sent[src] += current

            for hop_index, node in enumerate(
                path[:-1]
            ):

                node = int(node)

                path_hits[node] += 1

                received[node] += current

                attack = active_attack(
                    node,
                    float(time_s),
                    black,
                    grey,
                    worm_nodes
                )

                if attack == "blackhole":
                    p_drop = 0.95

                elif attack == "greyhole":
                    p_drop = 0.50

                elif attack == "wormhole":
                    p_drop = 0.15

                else:
                    p_drop = 0.02

                loss = int(
                    rng.binomial(
                        current,
                        p_drop
                    )
                )

                dropped[node] += loss

                keep = current - loss

                forwarded[node] += keep

                if attack == "wormhole":

                    hop_delay = rng.uniform(
                        2.0,
                        6.0
                    )

                else:

                    hop_delay = (
                        rng.uniform(
                            8.0,
                            25.0
                        )
                        + 20.0 * p_drop
                    )

                delay_sum[node] += hop_delay

                delay_count[node] += 1

                current = keep

                if current <= 0:
                    break

            if current > 0:

                received[
                    int(dst)
                ] += current

        degree = np.array(
            [
                len(x)
                for x in adj
            ],
            dtype=float
        )

        drop_rate = np.divide(
            dropped,
            received,
            out=np.zeros(
                n,
                dtype=float
            ),
            where=received > 0
        )

        forward_ratio = np.divide(
            forwarded,
            received,
            out=np.ones(
                n,
                dtype=float
            ),
            where=received > 0
        )

        avg_delay = np.divide(
            delay_sum,
            delay_count,
            out=np.full(
                n,
                15.0
            ),
            where=delay_count > 0
        )

        cumulative_work += (
            sent
            + received
            + forwarded
        )

        energy = np.clip(
            base_energy
            - 0.00002 * cumulative_work,
            0.05,
            1.0
        )

        route_pressure = (
            path_hits
            / max(
                1,
                path_hits.max()
            )
        )

        trust = (
            1.0
            - 0.75 * drop_rate
            - 0.10 * route_pressure
        )

        for node in worm_nodes:

            if time_s >= 300:
                trust[node] -= 0.25

        trust = np.clip(
            trust
            + rng.normal(
                0,
                0.025,
                size=n
            ),
            0.0,
            1.0
        )

        for node in range(n):

            attack = active_attack(
                node,
                float(time_s),
                black,
                grey,
                worm_nodes
            )

            label = (
                0
                if attack == "normal"
                else 1
            )

            node_rows.append([
                run_id,
                time_s,
                node,
                energy[node],
                trust[node],
                drop_rate[node],
                speed[node],
                forward_ratio[node],
                avg_delay[node],
                degree[node],
                int(received[node]),
                int(forwarded[node]),
                int(dropped[node]),
                int(path_hits[node]),
                label,
                attack
            ])

    nodes = pd.DataFrame(
        node_rows,
        columns=[
            "run",
            "time",
            "node_id",
            "energy",
            "trust",
            "drop_rate",
            "mobility",
            "forward_ratio",
            "delay",
            "degree",
            "packets_received",
            "packets_forwarded",
            "packets_dropped",
            "route_hits",
            "label",
            "attack"
        ]
    )

    edges = pd.DataFrame(
        edge_rows,
        columns=[
            "run",
            "time",
            "src",
            "dst",
            "edge_type",
            "distance"
        ]
    )

    return nodes, edges


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--positions",
        nargs="+",
        required=True,
        help="One or more ns-3 positions CSV files"
    )

    ap.add_argument(
        "--out-nodes",
        default="data/processed/nodes_dynamic.csv"
    )

    ap.add_argument(
        "--out-edges",
        default="data/processed/edges_dynamic.csv"
    )

    ap.add_argument(
        "--range",
        type=float,
        default=250.0,
        dest="comm_range"
    )

    ap.add_argument(
        "--flows",
        type=int,
        default=120
    )

    ap.add_argument(
        "--packets",
        type=int,
        default=20,
        dest="packets_per_flow"
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=2026
    )

    args = ap.parse_args()

    frames = [
        pd.read_csv(p)
        for p in args.positions
    ]

    all_pos = pd.concat(
        frames,
        ignore_index=True
    )

    required = {
        "run",
        "time",
        "node_id",
        "x",
        "y",
        "speed"
    }

    if not required.issubset(
        all_pos.columns
    ):

        raise ValueError(
            "positions CSV is missing columns: "
            + str(
                sorted(
                    required
                    - set(all_pos.columns)
                )
            )
        )

    node_parts = []
    edge_parts = []

    for run_id, rdf in all_pos.groupby(
        "run"
    ):

        nodes, edges = process_run(
            rdf,
            args.comm_range,
            args.flows,
            args.packets_per_flow,
            args.seed
        )

        node_parts.append(nodes)
        edge_parts.append(edges)

        print(
            f"Processed run {run_id}: "
            f"{len(nodes)} node-time rows, "
            f"{len(edges)} edges"
        )

    nodes = pd.concat(
        node_parts,
        ignore_index=True
    )

    edges = pd.concat(
        edge_parts,
        ignore_index=True
    )

    Path(
        args.out_nodes
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        args.out_edges
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    nodes.to_csv(
        args.out_nodes,
        index=False
    )

    edges.to_csv(
        args.out_edges,
        index=False
    )

    print(
        f"Saved {args.out_nodes}"
    )

    print(
        f"Saved {args.out_edges}"
    )


if __name__ == "__main__":
    main()

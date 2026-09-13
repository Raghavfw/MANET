import argparse
import pandas as pd

ap = argparse.ArgumentParser()

ap.add_argument(
    "--nodes",
    default="data/processed/nodes_dynamic.csv"
)

args = ap.parse_args()

df = pd.read_csv(args.nodes)

print("Columns:", list(df.columns))

print(
    "Runs:",
    sorted(df.run.unique())
)

print(
    "Time range:",
    df.time.min(),
    "to",
    df.time.max()
)

print("Nodes per run:")

print(
    df.groupby("run").node_id.nunique()
)

print("\nAttack counts:")

print(
    df.attack.value_counts()
)

for run_id, r in df.groupby("run"):

    times = sorted(
        r.time.unique()
    )

    cut = int(
        len(times) * 0.70
    )

    train_times = times[:cut]
    test_times = times[cut:]

    train = r[
        r.time.isin(train_times)
    ]

    test = r[
        r.time.isin(test_times)
    ]

    print(
        f"\nRun {run_id}: "
        f"{len(train_times)} train snapshots, "
        f"{len(test_times)} test snapshots"
    )

    print(
        "Train labels:",
        train.label.value_counts().to_dict()
    )

    print(
        "Test labels :",
        test.label.value_counts().to_dict()
    )

    if test.label.nunique() < 2:

        raise SystemExit(
            "ERROR: Test set does not contain "
            "both normal and malicious classes."
        )

print(
    "\nPASS: Every run's temporal test set contains both classes."
)


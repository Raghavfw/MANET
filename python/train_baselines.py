from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from xgboost import XGBClassifier


FEATURES = [
    "drop_rate",
    "trust",
    "forward_ratio",
    "energy"
]


def metrics(y, pred):

    p, r, f1, _ = (
        precision_recall_fscore_support(
            y,
            pred,
            labels=[1],
            average=None,
            zero_division=0
        )
    )

    return (
        accuracy_score(y, pred),
        float(p[0]),
        float(r[0]),
        float(f1[0])
    )


def split_temporal(run_df):

    times = sorted(
        run_df.time.unique()
    )

    cut = int(
        len(times) * 0.70
    )

    train = run_df[
        run_df.time.isin(
            times[:cut]
        )
    ]

    test = run_df[
        run_df.time.isin(
            times[cut:]
        )
    ]

    if test.label.nunique() < 2:

        raise ValueError(
            "Temporal test set must contain "
            "normal and malicious nodes"
        )

    return train, test


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--nodes",
        default="data/processed/nodes_dynamic.csv"
    )

    ap.add_argument(
        "--out",
        default="results/baseline_results.csv"
    )

    args = ap.parse_args()

    df = pd.read_csv(
        args.nodes
    )

    rows = []

    for run_id, r in df.groupby(
        "run"
    ):

        train, test = split_temporal(
            r
        )

        Xtr = train[
            FEATURES
        ].to_numpy()

        ytr = train[
            "label"
        ].to_numpy()

        Xte = test[
            FEATURES
        ].to_numpy()

        yte = test[
            "label"
        ].to_numpy()

        models = {

            "RandomForest":
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=int(run_id),
                    class_weight="balanced"
                ),

            "SVM":
                make_pipeline(
                    StandardScaler(),
                    SVC(
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced"
                    )
                ),

            "XGBoost":
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=int(run_id)
                ),

        }

        for name, model in models.items():

            model.fit(
                Xtr,
                ytr
            )

            pred = model.predict(
                Xte
            )

            acc, p, rec, f1 = metrics(
                yte,
                pred
            )

            rows.append([
                run_id,
                name,
                acc,
                p,
                rec,
                f1
            ])

            print(
                run_id,
                name,
                f"acc={acc:.4f}",
                f"mal_recall={rec:.4f}"
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "run",
            "model",
            "accuracy",
            "mal_precision",
            "mal_recall",
            "mal_f1"
        ]
    )

    Path(
        args.out
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.to_csv(
        args.out,
        index=False
    )

    summary = (
        out.groupby("model")
        [
            [
                "accuracy",
                "mal_precision",
                "mal_recall",
                "mal_f1"
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    print(
        "\nMEAN +/- STD OVER RUNS\n",
        summary
    )

    summary.to_csv(
        Path(args.out).with_name(
            "baseline_summary.csv"
        )
    )


if __name__ == "__main__":
    main()

    
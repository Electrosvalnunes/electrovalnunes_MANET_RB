#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explainable Bayesian IDS pipeline for FANET experiments using two datasets:
1) Source domain: Dataset_electrosvalnunes_manet_v2.csv (OMNeT++/AODV)
2) External domain: ns3_like_packet_loss_causes_v1_50k.csv (public NS-3-like dataset)

The external causal labels are mapped semantically as follows:
benign/mobility/interference -> Normal
congestion -> Flooding-like
malicious -> Blackhole-like

The external evaluation measures portability through semantic matching; it does
not assume perfect equivalence between the OMNeT++ attack scenarios and the NS-3
packet-loss causes.

Examples for protocol-specific analysis:
 python fanet_bn_xai_pipeline_english.py --protocol AODV --support 1500 --out AODV_results
 python fanet_bn_xai_pipeline_english.py --protocol OLSR --support 1500 --out OLSR_results
 python fanet_bn_xai_pipeline_english.py --protocol RPL --support 1500 --out RPL_results
 python fanet_bn_xai_pipeline_english.py --protocol DSR --support 1500 --out DSR_results
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mutual_info_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


# General settings

FEATURES = [
    "Node_Speed_ms",
    "Neighbor_Count",
    "PDR_Percentage",
    "E2E_Delay_ms",
    "Energy_Consumed_J",
    "Throughput_Kbps",
    "Queue_Drops",
    "Routing_Drops",
    "Control_Packets_Sent",
]

CLASSES = ["Blackhole", "Flooding", "Normal"]
STATES = ["Low", "Medium", "High"]


# Data preparation

def load_omnet_dataset(path: str) -> pd.DataFrame:
    """Load and validate the main OMNeT++ dataset."""
    df = pd.read_csv(path)
    required = ["Scenario"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the OMNeT++ dataset: {missing}")
    df = df.dropna(subset=required).copy()
    return df


def load_and_map_ns3_dataset(path: str, protocol_filter: str = "all") -> pd.DataFrame:
    """
    Load the external NS-3-like dataset and map its classes and features semantically.

    Notes:
    - Throughput_Kbps is approximated from pps * packet_size_bytes.
    - Routing_Drops is a semantic proxy based on cause_label == malicious.
    - Energy_Consumed_J is a proxy derived from PM.
    """
    raw = pd.read_csv(path)

    required = [
        "cause_label",
        "speed_mps",
        "neighbor_count",
        "delay_ms",
        "queue_drops",
        "pps",
        "packet_size_bytes",
        "PRR",
        "mac_tx_attempts",
        "PM",
        "routing",
        "drop_reason",
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in the NS-3 dataset: {missing}")

    if protocol_filter and protocol_filter.lower() != "all":
        raw = raw[raw["routing"].astype(str).str.upper() == protocol_filter.upper()].copy()

    scenario_map = {
        "benign": "Normal",
        "mobility": "Normal",
        "interference": "Normal",
        "congestion": "Flooding",
        "malicious": "Blackhole",
    }

    df = pd.DataFrame(index=raw.index)
    df["Scenario"] = raw["cause_label"].map(scenario_map)

    df["Node_Speed_ms"] = raw["speed_mps"]
    df["Neighbor_Count"] = raw["neighbor_count"]
    df["PDR_Percentage"] = raw["PRR"] * 100.0
    df["E2E_Delay_ms"] = raw["delay_ms"]

    # Energy proxy: lower PM values indicate higher estimated relative consumption.
    df["Energy_Consumed_J"] = (1.0 - raw["PM"]) * 10.0

    # Throughput proxy in Kbps based on packets per second and average packet size.
    df["Throughput_Kbps"] = raw["pps"] * raw["packet_size_bytes"] * 8.0 / 1000.0

    df["Queue_Drops"] = raw["queue_drops"]

    # Proxy for routing or malicious forwarding drops.
    # Values are kept as 100/0 to support comparable discretization with OMNeT++.
    df["Routing_Drops"] = np.where(raw["cause_label"] == "malicious", 100.0, 0.0)

    # Control-overhead proxy.
    df["Control_Packets_Sent"] = raw["mac_tx_attempts"]

    # Metadata retained for traceability.
    df["routing"] = raw["routing"].astype(str)
    df["cause_label_original"] = raw["cause_label"].astype(str)
    df["drop_reason"] = raw["drop_reason"].astype(str)

    df = df.dropna(subset=["Scenario"] + FEATURES).copy()
    return df


def balance_by_class(df: pd.DataFrame, target_size: int, random_state: int = 42) -> Tuple[pd.DataFrame, int]:
    """Balance the dataset by class, using the smallest available class when needed."""
    available = {c: int((df["Scenario"] == c).sum()) for c in CLASSES}
    n = min(target_size, min(available.values()))
    if n <= 0:
        raise ValueError(f"Insufficient samples for class balancing. Counts: {available}")

    parts = []
    for c in CLASSES:
        parts.append(df[df["Scenario"] == c].sample(n=n, random_state=random_state))
    balanced = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced, n

# Discretization

def fit_source_quantile_bins(train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Fit quantile bins on the source-domain training set only."""
    bins = {}
    for col in FEATURES:
        _, b = pd.qcut(train_df[col], q=3, retbins=True, duplicates="drop")

        if len(b) < 4:
            # Fallback for features with limited variability
            min_v, max_v = train_df[col].min(), train_df[col].max()
            if min_v == max_v:
                b = np.array([-np.inf, min_v, min_v + 1e-9, np.inf])
            else:
                b = np.linspace(min_v, max_v, 4)

        b = np.array(b, dtype=float)
        b[0] = -np.inf
        b[-1] = np.inf
        bins[col] = np.unique(b)

    return bins


def fit_target_percentile_bins_unlabeled(target_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Fit adaptive bins in the target domain using feature distributions only."""

    bins = {}
    for col in FEATURES:
        q33, q66 = target_df[col].quantile([0.33, 0.66]).values
        b = np.array([-np.inf, q33, q66, np.inf], dtype=float)

        if len(np.unique(b)) < 4:
            _, b2 = pd.qcut(target_df[col], q=3, retbins=True, duplicates="drop")
            b2 = np.array(b2, dtype=float)
            b2[0] = -np.inf
            b2[-1] = np.inf
            b = np.unique(b2)

        bins[col] = np.unique(b)

    return bins


def discretize(df: pd.DataFrame, bins: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Apply Low/Medium/High discretization using the supplied bins."""
    out = df.copy()
    for col in FEATURES:
        b = bins[col]
        labels = STATES[: len(b) - 1]
        out[col] = pd.cut(out[col], bins=b, labels=labels, include_lowest=True).astype(str)
    return out

# Discrete Bayesian classifier

@dataclass
class DiscreteBayesianIDS:
    """
    Simplified discrete Bayesian network with the following structure:

    Scenario -> PDR_Percentage
    Scenario -> E2E_Delay_ms
    Scenario -> Throughput_Kbps
    Scenario -> Queue_Drops
    Scenario -> Routing_Drops
    Scenario -> Control_Packets_Sent
    Scenario -> Energy_Consumed_J
    Node_Speed_ms -> PDR_Percentage
    Neighbor_Count -> PDR_Percentage

    During inference, Node_Speed_ms and Neighbor_Count are used as contextual
    evidence in the PDR conditional probability table.
    """

    alpha: float = 1.0

    def fit(self, df: pd.DataFrame) -> "DiscreteBayesianIDS":
        alpha = self.alpha

        counts = df["Scenario"].value_counts().reindex(CLASSES, fill_value=0).astype(float) + alpha
        self.log_prior_ = np.log((counts / counts.sum()).values)

        self.cond_logprob_ = {}
        for col in FEATURES:
            if col == "PDR_Percentage":
                continue

            tab = pd.crosstab(df["Scenario"], df[col])
            tab = tab.reindex(index=CLASSES, columns=STATES, fill_value=0).astype(float) + alpha
            prob = tab.div(tab.sum(axis=1), axis=0)

            self.cond_logprob_[col] = {}
            for scenario in CLASSES:
                for state in STATES:
                    self.cond_logprob_[col][(scenario, state)] = float(np.log(prob.loc[scenario, state]))

        # PDR conditioned on Scenario, Node_Speed_ms, and Neighbor_Count.
        grouped = df.groupby(
            ["Scenario", "Node_Speed_ms", "Neighbor_Count", "PDR_Percentage"],
            observed=False,
        ).size()

        self.pdr_logprob_ = {}
        for scenario in CLASSES:
            for speed_state in STATES:
                for neighbor_state in STATES:
                    raw_counts = np.array(
                        [
                            grouped.get((scenario, speed_state, neighbor_state, pdr_state), 0)
                            for pdr_state in STATES
                        ],
                        dtype=float,
                    )
                    smoothed = raw_counts + alpha
                    probs = smoothed / smoothed.sum()
                    for i, pdr_state in enumerate(STATES):
                        self.pdr_logprob_[(scenario, speed_state, neighbor_state, pdr_state)] = float(
                            np.log(probs[i])
                        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xs = {col: X[col].astype(str).values for col in FEATURES}
        proba = np.zeros((len(X), len(CLASSES)), dtype=float)

        for idx in range(len(X)):
            log_scores = self.log_prior_.copy()

            for class_idx, scenario in enumerate(CLASSES):
                for col in FEATURES:
                    if col == "PDR_Percentage":
                        continue
                    state = Xs[col][idx]
                    log_scores[class_idx] += self.cond_logprob_[col].get(
                        (scenario, state), np.log(1.0 / 3.0)
                    )

                key = (
                    scenario,
                    Xs["Node_Speed_ms"][idx],
                    Xs["Neighbor_Count"][idx],
                    Xs["PDR_Percentage"][idx],
                )
                log_scores[class_idx] += self.pdr_logprob_.get(key, np.log(1.0 / 3.0))

            shifted = np.exp(log_scores - np.max(log_scores))
            proba[idx, :] = shifted / shifted.sum()

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array(CLASSES)[np.argmax(proba, axis=1)]


# Evaluation and visualization

def evaluate_predictions(y_true, y_pred, y_proba=None) -> Dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    auc_macro = None
    auc_by_class = {}
    if y_proba is not None:
        try:
            y_idx = np.array([CLASSES.index(v) for v in y_true])
            y_bin = label_binarize(y_idx, classes=[0, 1, 2])
            auc_macro = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
            for i, c in enumerate(CLASSES):
                auc_by_class[c] = roc_auc_score(y_bin[:, i], y_proba[:, i])
        except Exception:
            auc_macro = None

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "auc_macro": auc_macro,
        "auc_by_class": auc_by_class,
    }


def save_metrics_csv(results: Dict[str, Dict], out_dir: Path) -> None:
    rows = []
    for env_name, res in results.items():
        report = res["classification_report"]
        for c in CLASSES:
            rows.append(
                {
                    "Environment": env_name,
                    "Class": c,
                    "Precision": report[c]["precision"],
                    "Recall": report[c]["recall"],
                    "F1-Score": report[c]["f1-score"],
                    "Support": report[c]["support"],
                    "Global_Accuracy": res["accuracy"],
                    "Macro_AUC": res["auc_macro"],
                    "Class_AUC": res["auc_by_class"].get(c, None),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "final_metrics.csv", index=False)


def save_confusion_matrices(results: Dict[str, Dict], out_dir: Path) -> None:
    for env_name, res in results.items():
        cm_df = pd.DataFrame(res["confusion_matrix"], index=CLASSES, columns=CLASSES)
        cm_df.to_csv(out_dir / f"confusion_matrix_{env_name}.csv")


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true, y_proba, title: str, path: Path) -> None:
    y_idx = np.array([CLASSES.index(v) for v in y_true])
    y_bin = label_binarize(y_idx, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{c} (AUC={auc_val:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_metric_comparison(results: Dict[str, Dict], out_dir: Path) -> None:
    rows = []
    for env_name, res in results.items():
        report = res["classification_report"]
        for c in CLASSES:
            rows.append({"Environment": env_name, "Class": c, "Metric": "Precision", "Value": report[c]["precision"]})
            rows.append({"Environment": env_name, "Class": c, "Metric": "Recall", "Value": report[c]["recall"]})
            rows.append({"Environment": env_name, "Class": c, "Metric": "F1", "Value": report[c]["f1-score"]})

    df = pd.DataFrame(rows)
    for c in CLASSES:
        subset = df[df["Class"] == c]
        pivot = subset.pivot(index="Metric", columns="Environment", values="Value")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Metrics by environment: {c}")
        ax.set_ylabel("Value")
        ax.set_xlabel("")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"comparative_metrics_{c}.png", dpi=300)
        plt.close(fig)


def compute_and_save_mi(
    omnet_disc_all: pd.DataFrame,
    ns3_disc_adapt: pd.DataFrame,
    out_dir: Path,
) -> None:
    rows = []
    for col in FEATURES:
        rows.append(
            {
                "Feature": col,
                "Environment": "OMNeT++",
                "Mutual_Information": mutual_info_score(omnet_disc_all["Scenario"], omnet_disc_all[col]),
            }
        )
        rows.append(
            {
                "Feature": col,
                "Environment": "NS-3 Adaptive",
                "Mutual_Information": mutual_info_score(ns3_disc_adapt["Scenario"], ns3_disc_adapt[col]),
            }
        )

    df_mi = pd.DataFrame(rows)
    df_mi.to_csv(out_dir / "global_mutual_information.csv", index=False)

    pivot = df_mi.pivot(index="Feature", columns="Environment", values="Mutual_Information")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mutual Information between Features and Class")
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "global_mutual_information.png", dpi=300)
    plt.close(fig)


def save_mapping_report(
    ns3_all: pd.DataFrame,
    ns3_balanced: pd.DataFrame,
    source_df: pd.DataFrame,
    support_used: int,
    protocol_filter: str,
    out_dir: Path,
) -> None:
    report = {
        "source_dataset_rows": int(len(source_df)),
        "external_dataset_rows_after_filter": int(len(ns3_all)),
        "protocol_filter": protocol_filter,
        "balanced_support_per_class": int(support_used),
        "source_classes": source_df["Scenario"].value_counts().to_dict(),
        "mapped_external_classes_before_balancing": ns3_all["Scenario"].value_counts().to_dict(),
        "mapped_external_classes_after_balancing": ns3_balanced["Scenario"].value_counts().to_dict(),
        "external_protocols_before_balancing": ns3_all["routing"].value_counts().to_dict(),
        "semantic_mapping": {
            "benign": "Normal",
            "mobility": "Normal",
            "interference": "Normal",
            "congestion": "Flooding-like",
            "malicious": "Blackhole-like",
        },
        "note": (
            "The external mapping is semantic and approximate. "
            "congestion represents Flooding-like behavior; malicious represents Blackhole-like or malicious forwarding behavior."
        ),
    }

    with open(out_dir / "mapping_and_support_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# Main execution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnet", default="Dataset_electrosvalnunes_manet_v2.csv")
    parser.add_argument("--ns3", default="ns3_like_packet_loss_causes_v1_50k.csv")
    parser.add_argument("--out", default="bn_xai_results")
    parser.add_argument(
        "--protocol",
        default="all",
        help="Use 'all' for the complete external dataset or select a protocol, e.g., OLSR, AODV, RPL, or DSR.",
    )
    parser.add_argument("--support", type=int, default=2400)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df_omnet = load_omnet_dataset(args.omnet)
    df_ns3_all = load_and_map_ns3_dataset(args.ns3, protocol_filter=args.protocol)

    # 2) Split the OMNeT++ source data
    train_src, test_src = train_test_split(
        df_omnet,
        test_size=0.30,
        stratify=df_omnet["Scenario"],
        random_state=args.random_state,
    )

    # 3) Fit and apply source-domain discretization
    source_bins = fit_source_quantile_bins(train_src)
    train_src_disc = discretize(train_src, source_bins)
    test_src_disc = discretize(test_src, source_bins)
    omnet_all_disc = discretize(df_omnet, source_bins)

    # 4) Train the model
    model = DiscreteBayesianIDS(alpha=1.0).fit(train_src_disc)

    # 5) Internal evaluation
    y_omnet = test_src_disc["Scenario"].values
    pred_omnet = model.predict(test_src_disc[FEATURES])
    proba_omnet = model.predict_proba(test_src_disc[FEATURES])

    # 6) Balance the external domain
    df_ns3_balanced_raw, support_used = balance_by_class(
        df_ns3_all,
        target_size=args.support,
        random_state=args.random_state,
    )

    # 7) External evaluation with source-domain thresholds
    ns3_static_disc = discretize(df_ns3_balanced_raw, source_bins)
    y_ns3_static = ns3_static_disc["Scenario"].values
    pred_ns3_static = model.predict(ns3_static_disc[FEATURES])
    proba_ns3_static = model.predict_proba(ns3_static_disc[FEATURES])

    # 8) External evaluation with unsupervised adaptive thresholds
    adaptive_bins = fit_target_percentile_bins_unlabeled(df_ns3_all)
    ns3_adapt_disc = discretize(df_ns3_balanced_raw, adaptive_bins)
    y_ns3_adapt = ns3_adapt_disc["Scenario"].values
    pred_ns3_adapt = model.predict(ns3_adapt_disc[FEATURES])
    proba_ns3_adapt = model.predict_proba(ns3_adapt_disc[FEATURES])

    # 9) Metrics
    results = {
        "OMNeT_internal": evaluate_predictions(y_omnet, pred_omnet, proba_omnet),
        "NS3_static": evaluate_predictions(y_ns3_static, pred_ns3_static, proba_ns3_static),
        "NS3_adaptive": evaluate_predictions(y_ns3_adapt, pred_ns3_adapt, proba_ns3_adapt),
    }

    save_metrics_csv(results, out_dir)
    save_confusion_matrices(results, out_dir)
    save_mapping_report(df_ns3_all, df_ns3_balanced_raw, df_omnet, support_used, args.protocol, out_dir)

    # 10) Figures
    plot_confusion_matrix(
        results["OMNeT_internal"]["confusion_matrix"],
        "Confusion Matrix - Internal OMNeT++",
        out_dir / "confusion_matrix_omnet.png",
    )
    plot_confusion_matrix(
        results["NS3_static"]["confusion_matrix"],
        "Confusion Matrix - NS-3 Static Threshold",
        out_dir / "confusion_matrix_ns3_static.png",
    )
    plot_confusion_matrix(
        results["NS3_adaptive"]["confusion_matrix"],
        "Confusion Matrix - NS-3 Adaptive Threshold",
        out_dir / "confusion_matrix_ns3_adaptive.png",
    )

    plot_roc_curves(y_omnet, proba_omnet, "ROC - Internal OMNeT++", out_dir / "roc_omnet.png")
    plot_roc_curves(y_ns3_static, proba_ns3_static, "ROC - NS-3 Static Threshold", out_dir / "roc_ns3_static.png")
    plot_roc_curves(y_ns3_adapt, proba_ns3_adapt, "ROC - NS-3 Adaptive Threshold", out_dir / "roc_ns3_adaptive.png")

    plot_metric_comparison(results, out_dir)
    compute_and_save_mi(omnet_all_disc, ns3_adapt_disc, out_dir)

    # 11) Console summary
    print("\nPIPELINE COMPLETED")
    print(f"Results saved to: {out_dir.resolve()}")
    print(f"External protocol filter: {args.protocol}")
    print(f"Balanced NS-3 support per class: {support_used}")

    for env_name, res in results.items():
        print(f"\n[{env_name}]")
        print(f"Accuracy: {res['accuracy']:.4f}")
        if res["auc_macro"] is not None:
            print(f"AUC macro: {res['auc_macro']:.4f}")
        print(pd.DataFrame(res["classification_report"]).T.round(4))

    print("\nMain files:")
    print("- final_metrics.csv")
    print("- global_mutual_information.csv")
    print("- mapping_and_support_report.json")
    print("- confusion_matrix_*.png")
    print("- roc_*.png")
    print("- comparative_metrics_*.png")



if __name__ == "__main__":
    main()

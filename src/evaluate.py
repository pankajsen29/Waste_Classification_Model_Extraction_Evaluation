#########################################################################################
# Surrogate vs Target Evaluation
# Metrics:
#   - Fidelity
#   - KL Divergence
#   - Mean Squared Error (MSE)
#   - Root Mean Squared Error (RMSE)
#########################################################################################

import json
import numpy as np
from scipy.special import rel_entr


# Load JSONL predictions
def load_jsonl(filepath):
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


# Fidelity = percentage of queries where surrogate predicts exactly the same class as the target.
# 1.0 = perfect extraction
# 0.95 = 95% agreement
def compute_fidelity(target_results, surrogate_results):

    matches = 0
    total = len(target_results)

    for tgt, sur in zip(target_results, surrogate_results):

        if tgt["image"] != sur["image"]:
            raise ValueError(
                f"Image mismatch: {tgt['image']} vs {sur['image']}"
            )

        if tgt["predicted_class"] == sur["predicted_class"]:
            matches += 1

    fidelity = matches / total
    return fidelity


# KL Divergence = Measures how closely the surrogate's probability distribution matches the target's.
# Lower is better.
# 0.0 = identical distributions.
def compute_average_kl_divergence(target_results, surrogate_results):

    kl_values = []

    for tgt, sur in zip(target_results, surrogate_results):

        target_scores = tgt["scores"]
        surrogate_scores = sur["scores"]

        classes = sorted(target_scores.keys())

        p = np.array([target_scores[c] for c in classes], dtype=np.float64)
        q = np.array([surrogate_scores[c] for c in classes], dtype=np.float64)

        # numerical stability
        eps = 1e-10
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)

        p = p / p.sum()
        q = q / q.sum()

        kl = np.sum(rel_entr(p, q))
        kl_values.append(kl)

    return np.mean(kl_values)


# MSE (Mean Squared Error): (standard metric)
# RMSE (Root Mean Squared Error): (more explainable metric)
# Lower is better.
# 0.0 = identical scores.
def compute_average_mse_rmse(target_results, surrogate_results):

    target_matrix = []
    surrogate_matrix = []

    for tgt, sur in zip(target_results, surrogate_results):

        classes = sorted(tgt["scores"].keys())

        target_matrix.append([tgt["scores"][c] for c in classes])
        surrogate_matrix.append([sur["scores"][c] for c in classes])

    target_matrix = np.array(target_matrix)
    surrogate_matrix = np.array(surrogate_matrix)

    mse = np.mean((target_matrix - surrogate_matrix) ** 2)

    return mse, np.sqrt(mse)


# Full evaluation
def evaluate_extraction(
        target_file="data/Query_Results/target_results.jsonl",
        surrogate_file="data/Query_Results/surrogate_results.jsonl"
):

    target_results = load_jsonl(target_file)
    surrogate_results = load_jsonl(surrogate_file)

    if len(target_results) != len(surrogate_results):
        raise ValueError("Target and surrogate result files have different lengths.")

    fidelity = compute_fidelity(target_results, surrogate_results)

    avg_kl = compute_average_kl_divergence(target_results, surrogate_results)

    avg_mse, avg_rmse = compute_average_mse_rmse(target_results, surrogate_results)

    print("\nWaste Classification Model Extraction Evaluation")
    print("-" * 40)
    print(f"Fidelity      : {fidelity:.4f}")
    print(f"Avg KL Div    : {avg_kl:.6f}")
    print(f"Avg MSE       : {avg_mse:.6f}")
    print(f"Avg RMSE      : {avg_rmse:.6f}")

    return {
        "fidelity": fidelity,
        "kl_divergence": avg_kl,
        "mse": avg_mse,
        "rmse": avg_rmse
    }
    
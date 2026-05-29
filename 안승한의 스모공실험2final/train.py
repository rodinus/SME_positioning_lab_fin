"""
train.py

Training script for the final HGB-Gate submission.

Final algorithm:
    Anchor-wise Calibration
    + Huber IRWLS initialization
    + Rich Feature Gated HistGradientBoosting residual correction

This script reproduces the fixed validation experiment, then trains the final
model on all labeled users and saves model.pkl.
"""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from main import HuberIRWLSSolver, find_mat_file, make_rich_features


HUBER_C = 1.345
HUBER_ITER = 5
HUBER_TOL = 1e-4
GATE_SCALE = 1.1


def hgb_params():
    return dict(
        loss="squared_error",
        learning_rate=0.02,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=2.0,
        max_bins=64,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        random_state=42,
    )


def compute_true_distances(p_user, bs_positions):
    diff = p_user[:, np.newaxis, :] - bs_positions[:, :, np.newaxis]
    return np.sqrt(np.sum(diff ** 2, axis=0))


def eval_metrics(p_hat, p_true):
    errors = np.linalg.norm(p_hat - p_true, axis=0)
    return {
        "MAE (m)": float(np.mean(errors)),
        "RMSE (m)": float(np.sqrt(np.mean(errors ** 2))),
        "Median Error (m)": float(np.median(errors)),
        "90% Error (m)": float(np.percentile(errors, 90)),
        "Max Error (m)": float(np.max(errors)),
    }


def batch_ls(d_input, bs_positions):
    solver = HuberIRWLSSolver(bs_positions)
    n = d_input.shape[1]
    p_ls = np.zeros((2, n), dtype=float)
    for i in range(n):
        p_ls[:, i] = solver.solve_ls(d_input[:, i])
    return p_ls


def batch_huber(d_cal, bs_positions):
    solver = HuberIRWLSSolver(bs_positions, huber_c=HUBER_C, max_iter=HUBER_ITER, tol=HUBER_TOL)
    num_anchor, num_user = d_cal.shape
    p_huber = np.zeros((2, num_user), dtype=float)
    residuals = np.zeros((num_anchor, num_user), dtype=float)
    weights = np.zeros((num_anchor, num_user), dtype=float)
    for i in range(num_user):
        pos, res, w = solver.solve_huber(d_cal[:, i])
        p_huber[:, i] = pos
        residuals[:, i] = res
        weights[:, i] = w
    return p_huber, residuals, weights


def run_pipeline(d_train, p_train, d_eval, p_eval, bs_positions):
    true_train = compute_true_distances(p_train, bs_positions)
    anchor_bias = np.mean(d_train - true_train, axis=1)

    d_train_cal = np.maximum(d_train - anchor_bias[:, np.newaxis], 1e-6)
    d_eval_cal = np.maximum(d_eval - anchor_bias[:, np.newaxis], 1e-6)

    p_raw = batch_ls(d_eval, bs_positions)
    p_cal = batch_ls(d_eval_cal, bs_positions)

    p_huber_train, res_train, w_train = batch_huber(d_train_cal, bs_positions)
    p_huber_eval, res_eval, w_eval = batch_huber(d_eval_cal, bs_positions)

    X_train = make_rich_features(d_train, d_train_cal, p_huber_train, res_train, w_train, bs_positions)
    y_train = (p_train - p_huber_train).T
    X_eval = make_rich_features(d_eval, d_eval_cal, p_huber_eval, res_eval, w_eval, bs_positions)

    hgb = MultiOutputRegressor(HistGradientBoostingRegressor(**hgb_params()))
    hgb.fit(X_train, y_train)

    delta = hgb.predict(X_eval).T
    p_final = p_huber_eval + GATE_SCALE * delta

    rows = [
        {"Method": "Raw LS", **eval_metrics(p_raw, p_eval)},
        {"Method": "Anchor-wise Calibrated LS", **eval_metrics(p_cal, p_eval)},
        {"Method": "Calibrated + Huber IRWLS", **eval_metrics(p_huber_eval, p_eval)},
        {"Method": "Final Rich Feature Gated HGB", **eval_metrics(p_final, p_eval)},
    ]
    return pd.DataFrame(rows)


def validation_experiment(d_hat, p, bs_positions):
    train_idx, val_idx = train_test_split(
        np.arange(d_hat.shape[1]), test_size=0.2, random_state=42, shuffle=True
    )
    d_train, p_train = d_hat[:, train_idx], p[:, train_idx]
    d_val, p_val = d_hat[:, val_idx], p[:, val_idx]
    return run_pipeline(d_train, p_train, d_val, p_val, bs_positions)


def train_final_model(d_hat, p, bs_positions):
    true_dist = compute_true_distances(p, bs_positions)
    anchor_bias = np.mean(d_hat - true_dist, axis=1)
    d_cal = np.maximum(d_hat - anchor_bias[:, np.newaxis], 1e-6)

    p_huber, residuals, weights = batch_huber(d_cal, bs_positions)
    X = make_rich_features(d_hat, d_cal, p_huber, residuals, weights, bs_positions)
    y = (p - p_huber).T

    hgb = MultiOutputRegressor(HistGradientBoostingRegressor(**hgb_params()))
    hgb.fit(X, y)

    return {
        "algorithm": "Anchor-wise Calibration + Huber IRWLS + Rich Feature Gated HGB residual correction",
        "anchor_bias": anchor_bias,
        "hgb_model": hgb,
        "hgb_params": hgb_params(),
        "gate_scale": GATE_SCALE,
        "huber_c": HUBER_C,
        "huber_iter": HUBER_ITER,
        "huber_tol": HUBER_TOL,
        "feature_set": "rich",
        "feature_layout": [
            "raw RTT distance",
            "calibrated RTT distance",
            "Huber initial position",
            "anchor residual",
            "absolute residual",
            "residual divided by calibrated distance",
            "Huber final weight",
            "predicted anchor distance from Huber position",
            "residual and weight statistics",
        ],
    }


def main():
    start = time.time()
    mat_path = find_mat_file()
    data = sio.loadmat(mat_path, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)
    if "BS_positions" in data:
        bs_positions = np.asarray(data["BS_positions"], dtype=float)
    else:
        bs_positions = np.asarray(data["p_bs"], dtype=float)

    print("Running validation experiment...")
    results = validation_experiment(d_hat, p, bs_positions)
    print(results.to_string(index=False))
    results.to_csv("validation_results.csv", index=False)

    print("\nTraining final HGB-Gate model with all labeled users...")
    model_data = train_final_model(d_hat, p, bs_positions)
    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("Saved model.pkl")
    print("Saved validation_results.csv")
    print(f"Elapsed time: {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()

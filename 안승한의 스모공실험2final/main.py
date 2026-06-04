"""
main.py

Fixed final submission entrypoint.
This version reads gate_scale from model.pkl, while also supporting older
model.pkl files that may contain gate_config.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares


def find_mat_file() -> str:
    candidates = ["DH_FR1.mat", "InF_DH_FR1.mat", "_InF_DH_FR1.mat", "InF_DH_FR1_val.mat"]
    for name in candidates:
        if os.path.exists(name):
            return name
    mat_files = sorted(Path(".").glob("*.mat"))
    if mat_files:
        return str(mat_files[0])
    raise FileNotFoundError("No .mat file found. Expected DH_FR1.mat.")


class HuberIRWLSSolver:
    def __init__(self, bs_positions: np.ndarray, huber_c: float = 1.345, max_iter: int = 5, tol: float = 1e-4):
        self.bs_positions = np.asarray(bs_positions, dtype=float)
        self.huber_c = float(huber_c)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.lower_bound = np.array([
            self.bs_positions[0].min() - 20.0,
            self.bs_positions[1].min() - 20.0,
        ])
        self.upper_bound = np.array([
            self.bs_positions[0].max() + 20.0,
            self.bs_positions[1].max() + 20.0,
        ])
        self.initial_point = np.mean(self.bs_positions, axis=1)

    def solve_ls(self, distance_vector: np.ndarray, weights=None, x0=None) -> np.ndarray:
        distance_vector = np.asarray(distance_vector, dtype=float).reshape(-1)
        if x0 is None:
            x0 = self.initial_point
        if weights is None:
            sqrt_w = np.ones_like(distance_vector)
        else:
            sqrt_w = np.sqrt(np.maximum(np.asarray(weights, dtype=float).reshape(-1), 1e-12))

        def residual(pos):
            predicted_distance = np.linalg.norm(self.bs_positions.T - pos.reshape(1, 2), axis=1)
            return sqrt_w * (predicted_distance - distance_vector)

        result = least_squares(
            residual,
            x0=x0,
            bounds=(self.lower_bound, self.upper_bound),
            max_nfev=200,
        )
        return result.x

    @staticmethod
    def robust_sigma_mad(residuals: np.ndarray) -> float:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma = 1.4826 * mad
        if sigma < 1e-6:
            sigma = np.std(residuals) + 1e-6
        return float(sigma)

    def huber_weights(self, residuals: np.ndarray) -> np.ndarray:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        sigma = self.robust_sigma_mad(residuals)
        u = residuals / sigma
        abs_u = np.abs(u)
        weights = np.ones_like(abs_u)
        mask = abs_u > self.huber_c
        weights[mask] = self.huber_c / (abs_u[mask] + 1e-12)
        return weights / (np.mean(weights) + 1e-12)

    def solve_huber(self, distance_vector: np.ndarray):
        distance_vector = np.asarray(distance_vector, dtype=float).reshape(-1)
        pos = self.solve_ls(distance_vector)
        for _ in range(self.max_iter):
            predicted_distance = np.linalg.norm(self.bs_positions.T - pos.reshape(1, 2), axis=1)
            residuals = predicted_distance - distance_vector
            weights = self.huber_weights(residuals)
            new_pos = self.solve_ls(distance_vector, weights=weights, x0=pos)
            if np.linalg.norm(new_pos - pos) < self.tol:
                pos = new_pos
                break
            pos = new_pos
        predicted_distance = np.linalg.norm(self.bs_positions.T - pos.reshape(1, 2), axis=1)
        final_residuals = predicted_distance - distance_vector
        final_weights = self.huber_weights(final_residuals)
        return pos, final_residuals, final_weights


def make_rich_features(d_raw, d_cal, p_huber, residuals, weights, bs_positions):
    abs_res = np.abs(residuals)
    res_over_d = residuals / (d_cal + 1e-6)
    num_user = d_cal.shape[1]
    pred_dist = np.zeros_like(d_cal)
    for j in range(num_user):
        pred_dist[:, j] = np.linalg.norm(
            bs_positions.T - p_huber[:, j].reshape(1, 2),
            axis=1,
        )
    stats = np.vstack([
        np.mean(abs_res, axis=0),
        np.median(abs_res, axis=0),
        np.max(abs_res, axis=0),
        np.std(residuals, axis=0),
        np.mean(weights, axis=0),
        np.min(weights, axis=0),
        np.max(weights, axis=0),
        np.std(weights, axis=0),
    ]).T
    return np.hstack([
        d_raw.T,
        d_cal.T,
        p_huber.T,
        residuals.T,
        abs_res.T,
        res_over_d.T,
        weights.T,
        pred_dist.T,
        stats,
    ])


def get_gate_scale(model_data: dict) -> float:
    if "gate_scale" in model_data:
        return float(model_data["gate_scale"])
    if "gate_config" in model_data and isinstance(model_data["gate_config"], dict):
        return float(model_data["gate_config"].get("global_scale", 1.1))
    return 1.1


def main():
    mat_path = find_mat_file()
    data = sio.loadmat(mat_path, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    if "BS_positions" in data:
        bs_positions = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        bs_positions = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("BS_positions or p_bs not found in .mat file.")

    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)

    anchor_bias = np.asarray(model_data["anchor_bias"], dtype=float).reshape(-1)
    hgb_model = model_data["hgb_model"]
    gate_scale = get_gate_scale(model_data)
    huber_c = float(model_data.get("huber_c", 1.345))
    huber_iter = int(model_data.get("huber_iter", 5))
    huber_tol = float(model_data.get("huber_tol", 1e-4))

    num_anchor, num_user = d_hat.shape
    if anchor_bias.shape[0] != num_anchor:
        raise ValueError("anchor_bias size does not match number of anchors.")

    d_cal = np.maximum(d_hat - anchor_bias[:, np.newaxis], 1e-6)
    solver = HuberIRWLSSolver(bs_positions, huber_c=huber_c, max_iter=huber_iter, tol=huber_tol)
    p_huber = np.zeros((2, num_user), dtype=float)
    residuals = np.zeros((num_anchor, num_user), dtype=float)
    weights = np.zeros((num_anchor, num_user), dtype=float)
    for u in range(num_user):
        pos, res, w = solver.solve_huber(d_cal[:, u])
        p_huber[:, u] = pos
        residuals[:, u] = res
        weights[:, u] = w

    X = make_rich_features(d_hat, d_cal, p_huber, residuals, weights, bs_positions)
    delta = hgb_model.predict(X).T
    p_hat = p_huber + gate_scale * delta
    return np.asarray(p_hat, dtype=float)


if __name__ == "__main__":
    result = main()
    print(result.shape)

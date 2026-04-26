from __future__ import annotations

from typing import Any

import numpy as np
import shap
from torch import nn

from app.agents.federated_learning.tabular.model import LogisticFraudModel, predict_probabilities


def compute_shap_summary(
    *,
    model: nn.Module,
    x_background: np.ndarray,
    x_explain: np.ndarray,
    feature_names: list[str],
    feature_owner_map: dict[str, str] | None = None,
    max_explain_samples: int = 128,
    top_k: int = 20,
    top_k_per_client: int = 10,
) -> dict[str, Any]:
    if x_background.shape[0] == 0 or x_explain.shape[0] == 0:
        return {
            "num_samples": 0,
            "explainer": "none",
            "global_mean_abs_shap": [],
            "top_features": [],
            "top_features_by_client": {},
        }

    background_count = min(100, x_background.shape[0])
    explain_count = min(max_explain_samples, x_explain.shape[0])

    background = x_background[:background_count]
    explain = x_explain[:explain_count]

    def predict_fn(values: np.ndarray) -> np.ndarray:
        return predict_probabilities(model, values)

    values, explainer_name = _compute_shap_values(
        model=model,
        predict_fn=predict_fn,
        background=background,
        explain=explain,
    )

    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(-mean_abs)

    global_mean_abs = [
        {
            "feature": feature_names[index],
            "mean_abs_shap": float(mean_abs[index]),
            "owner_client": (feature_owner_map or {}).get(feature_names[index], "unknown"),
        }
        for index in order
    ]

    top_features = [
        {
            "feature": feature_names[index],
            "mean_abs_shap": float(mean_abs[index]),
            "owner_client": (feature_owner_map or {}).get(feature_names[index], "unknown"),
        }
        for index in order[: min(top_k, len(feature_names))]
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    if feature_owner_map:
        for item in top_features:
            owner = item["owner_client"]
            grouped.setdefault(owner, []).append(item)
        grouped = {
            owner: items[:top_k_per_client]
            for owner, items in grouped.items()
        }

    return {
        "num_samples": int(explain_count),
        "explainer": explainer_name,
        "global_mean_abs_shap": global_mean_abs,
        "top_features": top_features,
        "top_features_by_client": grouped,
    }


def _compute_shap_values(
    *,
    model: nn.Module,
    predict_fn: Any,
    background: np.ndarray,
    explain: np.ndarray,
) -> tuple[np.ndarray, str]:
    if isinstance(model, LogisticFraudModel):
        try:
            weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
            bias = float(model.linear.bias.detach().cpu().numpy().reshape(-1)[0])
            explainer = shap.LinearExplainer((weights, bias), background)
            shap_values = explainer.shap_values(explain)
            return _as_2d_values(shap_values), "linear"
        except Exception:
            pass

    kernel_nsamples = min(200, max(20, explain.shape[1] * 5))
    kernel_explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = kernel_explainer.shap_values(explain, nsamples=kernel_nsamples)
    return _as_2d_values(shap_values), "kernel"


def _as_2d_values(raw_values: Any) -> np.ndarray:
    if hasattr(raw_values, "values"):
        raw_values = raw_values.values

    values = np.asarray(raw_values)
    if values.ndim == 1:
        return values.reshape(1, -1)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")

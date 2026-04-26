from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
flwr = pytest.importorskip("flwr", reason="flwr not installed — skipping federated tests")

from app.agents.federated_learning.tabular.data import TabularFederatedDataset
from app.agents.federated_learning.tabular.model import (
    LocalTrainingConfig,
    ModelConfig,
    create_model,
    model_to_ndarrays,
    ndarrays_to_model,
    train_local_model,
)
from app.agents.federated_learning.tabular.partitioning import (
    apply_vertical_mask,
    build_partition_metadata,
    build_vertical_partitions,
)


def test_vertical_partitioning_is_disjoint_and_masked() -> None:
    feature_names = [f"f{i}" for i in range(9)]
    partitions = build_vertical_partitions(feature_names, num_clients=3)

    combined: set[str] = set()
    for partition in partitions:
        owned = set(partition.owned_columns)
        assert not combined.intersection(owned)
        combined.update(owned)
    assert combined == set(feature_names)

    metadata = build_partition_metadata(
        partitions=partitions,
        feature_names=feature_names,
        num_train_rows=100,
        num_eval_rows=25,
    )
    assert metadata["row_index_policy"] == "shared_all_rows"
    assert metadata["shared_row_indices"]["train"]["count"] == 100
    assert metadata["shared_row_indices"]["eval"]["count"] == 25

    frame = pd.DataFrame(np.arange(27, dtype=np.float32).reshape(3, 9), columns=feature_names)
    owned_columns = partitions[0].owned_columns
    masked = apply_vertical_mask(frame, owned_columns)

    for column in feature_names:
        if column in owned_columns:
            assert np.allclose(masked[column].to_numpy(), frame[column].to_numpy())
        else:
            assert np.allclose(masked[column].to_numpy(), 0.0)


def test_local_training_updates_model_parameters() -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(128, 6)).astype(np.float32)
    y_train = ((x_train[:, 0] + 0.5 * x_train[:, 1]) > 0.0).astype(np.float32)

    model = create_model(input_dim=6, config=ModelConfig(model_type="logistic"))
    before = [array.copy() for array in model_to_ndarrays(model)]

    metrics = train_local_model(
        model,
        x_train,
        y_train,
        config=LocalTrainingConfig(epochs=3, batch_size=32, learning_rate=1e-2),
    )

    after = model_to_ndarrays(model)
    assert np.isfinite(metrics["train_loss"])
    assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_fedavg_parameter_roundtrip_is_compatible_across_three_clients() -> None:
    feature_names = [f"f{i}" for i in range(6)]
    partitions = build_vertical_partitions(feature_names, num_clients=3)

    frame = pd.DataFrame(np.random.default_rng(5).normal(size=(20, 6)), columns=feature_names)
    client_parameter_sets: list[list[np.ndarray]] = []

    for partition in partitions:
        masked = apply_vertical_mask(frame, partition.owned_columns)
        model = create_model(input_dim=masked.shape[1], config=ModelConfig(model_type="logistic"))
        params = model_to_ndarrays(model)
        client_parameter_sets.append(params)

    shapes = [[tuple(arr.shape) for arr in params] for params in client_parameter_sets]
    assert all(shape == shapes[0] for shape in shapes)

    averaged = [
        np.mean(np.stack([params[i] for params in client_parameter_sets], axis=0), axis=0)
        for i in range(len(client_parameter_sets[0]))
    ]

    global_model = create_model(input_dim=len(feature_names), config=ModelConfig(model_type="logistic"))
    ndarrays_to_model(global_model, averaged)
    restored = model_to_ndarrays(global_model)
    assert all(np.allclose(a, b) for a, b in zip(averaged, restored))


def test_federated_simulation_smoke_with_synthetic_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_flwr = types.SimpleNamespace(
        simulation=types.SimpleNamespace(start_simulation=lambda **kwargs: SimpleNamespace(
            losses_distributed=[(1, 0.5)],
            metrics_distributed_fit={"train_loss": [(1, 0.4)]},
            metrics_distributed={"auc_pr": [(1, 0.6)]},
            metrics_centralized={"loss": [(1, 0.45)]},
        )),
        server=types.SimpleNamespace(
            ServerConfig=lambda num_rounds: SimpleNamespace(num_rounds=num_rounds),
            history=types.SimpleNamespace(History=object),
        ),
    )
    fake_flwr_bridge = types.SimpleNamespace(
        build_client_fn=lambda **kwargs: (lambda cid: None),
        create_tracking_strategy=lambda *, initial_parameters, min_clients, evaluate_fn: SimpleNamespace(
            latest_parameters=[array.copy() for array in initial_parameters]
        ),
    )
    fake_shap = types.SimpleNamespace(
        Explainer=lambda *args, **kwargs: None,
        LinearExplainer=lambda *args, **kwargs: None,
        KernelExplainer=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "flwr", fake_flwr)
    monkeypatch.setitem(sys.modules, "app.agents.federated_learning.tabular.flwr_bridge", fake_flwr_bridge)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    if "app.agents.federated_learning.tabular.simulation" in sys.modules:
        del sys.modules["app.agents.federated_learning.tabular.simulation"]
    sim_mod = importlib.import_module("app.agents.federated_learning.tabular.simulation")

    feature_names = [f"f{i}" for i in range(6)]
    rng = np.random.default_rng(11)

    x_train = pd.DataFrame(rng.normal(size=(30, 6)).astype(np.float32), columns=feature_names)
    y_train = pd.Series((x_train["f0"] > 0).astype(int), name="isFraud")
    x_eval = pd.DataFrame(rng.normal(size=(12, 6)).astype(np.float32), columns=feature_names)
    y_eval = pd.Series((x_eval["f0"] > 0).astype(int), name="isFraud")

    dataset = TabularFederatedDataset(
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        feature_names=feature_names,
    )

    def fake_load_dataset(**_: object) -> TabularFederatedDataset:
        return dataset

    def fake_shap_summary(**_: object) -> dict[str, object]:
        return {
            "num_samples": 4,
            "explainer": "linear",
            "global_mean_abs_shap": [
                {"feature": "f0", "mean_abs_shap": 0.3, "owner_client": "client_0"}
            ],
            "top_features": [
                {"feature": "f0", "mean_abs_shap": 0.3, "owner_client": "client_0"}
            ],
            "top_features_by_client": {
                "client_0": [
                    {"feature": "f0", "mean_abs_shap": 0.3, "owner_client": "client_0"}
                ]
            },
        }

    monkeypatch.setattr(sim_mod, "load_ieee_cis_dataset", fake_load_dataset)
    monkeypatch.setattr(sim_mod, "compute_shap_summary", fake_shap_summary)

    result = sim_mod.run_federated_simulation(
        config=sim_mod.SimulationConfig(
            num_clients=3,
            rounds=1,
            local_epochs=1,
            model_type="logistic",
            enable_mlflow=False,
        ),
        output_dir=tmp_path,
    )

    assert "final_metrics" in result
    assert "aggregated_global_parameters" in result

    partition_file = tmp_path / "vertical_partition_map.json"
    shap_file = tmp_path / "shap_summary.json"
    metrics_file = tmp_path / "federated_metrics.json"

    assert partition_file.exists()
    assert shap_file.exists()
    assert metrics_file.exists()

    partition_data = json.loads(partition_file.read_text(encoding="utf-8"))
    assert partition_data["row_index_policy"] == "shared_all_rows"

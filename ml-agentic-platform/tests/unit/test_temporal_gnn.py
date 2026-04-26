from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data

from app.agents.graph_fraud.temporal_gnn.config import TemporalModelConfig, TemporalTrainingConfig
from app.agents.graph_fraud.temporal_gnn.data import build_temporal_node_classification_data
from app.agents.graph_fraud.temporal_gnn.model import TemporalGraphModel
from app.agents.graph_fraud.temporal_gnn.train import _parse_args
from app.agents.graph_fraud.temporal_gnn.trainer import _safe_auc_pr, train_and_evaluate


def test_build_temporal_node_classification_data_for_paysim(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "PAYMENT", "TRANSFER"],
            "amount": [100.0, 200.0, 150.0, 300.0],
            "nameOrig": ["C1", "C2", "C3", "C1"],
            "oldbalanceOrg": [500.0, 600.0, 700.0, 800.0],
            "newbalanceOrig": [400.0, 400.0, 550.0, 500.0],
            "nameDest": ["M1", "M2", "M3", "M2"],
            "oldbalanceDest": [0.0, 100.0, 200.0, 300.0],
            "newbalanceDest": [100.0, 300.0, 350.0, 600.0],
            "isFraud": [0, 1, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
        }
    )
    csv_path = tmp_path / "paysim_sample.csv"
    frame.to_csv(csv_path, index=False)

    data, config = build_temporal_node_classification_data(
        dataset_name="paysim",
        csv_path=csv_path,
    )

    assert config.dataset_name == "paysim"
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 5
    assert data.x.shape[0] == data.num_nodes
    assert int(data.y.sum().item()) >= 1
    assert data.train_mask.any().item()
    assert data.val_mask.any().item()
    assert data.test_mask.any().item()


@pytest.mark.parametrize("backbone", ["gcn", "gat"])
def test_temporal_graph_model_forward_shape(backbone: str) -> None:
    data = Data(
        x=torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.2, 0.3, 0.1],
                [2.0, 1.0, 3.0, 0.4, 0.5, 0.2],
                [1.0, 1.0, 2.0, 0.6, 0.7, 0.3],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[0.1], [0.3], [0.2]], dtype=torch.float32),
        edge_time=torch.tensor([0.1, 0.4, 0.8], dtype=torch.float32),
    )
    model = TemporalGraphModel(
        node_feature_dim=6,
        edge_attr_dim=1,
        config=TemporalModelConfig(
            backbone=backbone,
            hidden_dim=32,
            num_layers=2,
            time_encoding_dim=8,
            gat_heads=4,
            dropout=0.1,
        ),
    )

    logits = model(data)
    assert logits.shape == (3,)


def test_safe_auc_pr_returns_expected_values() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    probabilities = np.array([0.05, 0.95, 0.1, 0.9], dtype=np.float32)

    auc_pr = _safe_auc_pr(labels, probabilities)
    assert pytest.approx(auc_pr, rel=1e-6) == 1.0

    single_class_labels = np.array([0, 0, 0], dtype=np.float32)
    single_class_probabilities = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert _safe_auc_pr(single_class_labels, single_class_probabilities) == 0.0


# ─────────────────────────── Logging & CLI tests ─────────────────────────────


def _make_graph_data() -> Data:
    """Minimal synthetic PyG Data with both positive and negative nodes."""
    return Data(
        x=torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.2, 0.3, 0.1],
                [2.0, 1.0, 3.0, 0.4, 0.5, 0.2],
                [1.0, 1.0, 2.0, 0.6, 0.7, 0.3],
                [0.5, 1.5, 2.5, 0.1, 0.2, 0.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[0.1], [0.3], [0.2], [0.4]], dtype=torch.float32),
        edge_time=torch.tensor([0.1, 0.4, 0.8, 0.5], dtype=torch.float32),
        y=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32),
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )


def _make_tiny_model() -> TemporalGraphModel:
    return TemporalGraphModel(
        node_feature_dim=6,
        edge_attr_dim=1,
        config=TemporalModelConfig(
            backbone="gcn",
            hidden_dim=16,
            num_layers=1,
            time_encoding_dim=4,
            gat_heads=1,
            dropout=0.0,
        ),
    )


def test_parse_args_log_level_and_interval_defaults() -> None:
    args = _parse_args(["--dataset", "ieee_cis", "--ieee-csv-path", "dummy.csv"])
    assert args.log_level == "INFO"
    assert args.log_interval == 1


def test_parse_args_log_level_and_interval_overrides() -> None:
    args = _parse_args(
        [
            "--dataset", "ieee_cis",
            "--ieee-csv-path", "dummy.csv",
            "--log-level", "DEBUG",
            "--log-interval", "5",
        ]
    )
    assert args.log_level == "DEBUG"
    assert args.log_interval == 5


def test_trainer_logs_epoch_progress(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    data = _make_graph_data()
    model = _make_tiny_model()
    with caplog.at_level(logging.INFO, logger="app.agents.graph_fraud.temporal_gnn.trainer"):
        train_and_evaluate(
            model=model,
            data=data,
            config=TemporalTrainingConfig(epochs=3, patience=10, log_interval=1, device="cpu"),
        )

    epoch_logs = [r.message for r in caplog.records if "Epoch" in r.message]
    # epochs 1, 2, 3 should each produce exactly one progress line
    assert len(epoch_logs) >= 3


def test_trainer_logs_early_stopping(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    data = _make_graph_data()
    model = _make_tiny_model()
    with caplog.at_level(logging.INFO, logger="app.agents.graph_fraud.temporal_gnn.trainer"):
        train_and_evaluate(
            model=model,
            data=data,
            # patience=1 forces early stopping within the first few epochs
            config=TemporalTrainingConfig(epochs=50, patience=1, log_interval=1, device="cpu"),
        )

    early_stop_logs = [r.message for r in caplog.records if "Early stopping" in r.message]
    assert len(early_stop_logs) == 1


def test_trainer_log_interval_controls_frequency(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    data = _make_graph_data()
    model = _make_tiny_model()
    with caplog.at_level(logging.INFO, logger="app.agents.graph_fraud.temporal_gnn.trainer"):
        train_and_evaluate(
            model=model,
            data=data,
            # log_interval=5 on 10 epochs → logs epoch 1, 5, 10 (first, every 5, last)
            config=TemporalTrainingConfig(epochs=10, patience=20, log_interval=5, device="cpu"),
        )

    epoch_logs = [r.message for r in caplog.records if "Epoch" in r.message]
    # Should be well below 10 (one per epoch) when log_interval=5
    assert 1 <= len(epoch_logs) <= 4

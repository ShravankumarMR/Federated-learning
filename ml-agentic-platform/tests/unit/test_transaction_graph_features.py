import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.data_engineering.features.transaction_graph import GraphBuildConfig, build_pyg_data, preprocess_transactions


def test_preprocess_transactions_handles_missing_and_normalizes() -> None:
    frame = pd.DataFrame(
        {
            "source_account": ["A", None, "C"],
            "target_account": ["B", "D", None],
            "timestamp": [1, None, 3],
            "amount": [100.0, None, 300.0],
        }
    )

    processed = preprocess_transactions(
        frame,
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_cols=["amount"],
    )

    assert processed["source_account"].isna().sum() == 0
    assert processed["target_account"].isna().sum() == 0
    assert processed["timestamp"].isna().sum() == 0
    assert pytest.approx(float(processed["amount"].mean()), abs=1e-6) == 0.0


def test_build_pyg_data_shapes() -> None:
    frame = pd.DataFrame(
        {
            "source_account": ["A", "B", "A"],
            "target_account": ["B", "C", "C"],
            "timestamp": [1, 2, 3],
            "amount": [0.1, 0.2, 0.3],
            "isFraud": [0, 1, 0],
        }
    )
    config = GraphBuildConfig(
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_edge_cols=["amount"],
        label_col="isFraud",
    )

    data = build_pyg_data(frame, config)

    assert data.edge_index.shape == (2, 3)
    assert data.edge_attr.shape == (3, 1)
    assert data.edge_time.shape[0] == 3
    assert data.x.shape[1] == 2
    assert data.y.shape[0] == 3
    assert data.edge_index.dtype == torch.long

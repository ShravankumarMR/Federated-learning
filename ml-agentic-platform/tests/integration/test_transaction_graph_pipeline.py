from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from app.data_engineering.pipelines.transaction_graph_pipeline import build_transaction_graphs


def test_build_transaction_graphs_end_to_end(tmp_path: Path) -> None:
    ieee_path = tmp_path / "train_transaction.csv"
    paysim_path = tmp_path / "paysim.csv"

    pd.DataFrame(
        {
            "TransactionDT": [1000, 1100, 1200],
            "TransactionAmt": [50.0, 65.0, 10.0],
            "card1": [1111, 1111, 2222],
            "card2": [100, 100, 200],
            "addr1": [10, 10, 20],
            "P_emaildomain": ["a.com", "a.com", "b.com"],
            "card4": ["visa", "visa", "mastercard"],
            "addr2": [50, 50, 60],
            "R_emaildomain": ["x.com", "y.com", "x.com"],
            "ProductCD": ["W", "W", "C"],
            "isFraud": [0, 1, 0],
        }
    ).to_csv(ieee_path, index=False)

    pd.DataFrame(
        {
            "step": [1, 2, 3],
            "amount": [100.0, 250.0, 10.0],
            "nameOrig": ["C1", "C2", "C1"],
            "oldbalanceOrg": [500.0, 800.0, 500.0],
            "newbalanceOrig": [400.0, 550.0, 490.0],
            "nameDest": ["M1", "M2", "M2"],
            "oldbalanceDest": [0.0, 20.0, 20.0],
            "newbalanceDest": [100.0, 270.0, 30.0],
            "isFraud": [0, 1, 0],
        }
    ).to_csv(paysim_path, index=False)

    graphs = build_transaction_graphs(ieee_path, paysim_path)

    assert set(graphs.keys()) == {"ieee_cis", "paysim"}
    assert graphs["ieee_cis"].edge_index.shape[1] == 3
    assert graphs["paysim"].edge_index.shape[1] == 3
    assert graphs["ieee_cis"].edge_attr.shape[0] == graphs["ieee_cis"].edge_index.shape[1]
    assert graphs["paysim"].edge_attr.shape[0] == graphs["paysim"].edge_index.shape[1]

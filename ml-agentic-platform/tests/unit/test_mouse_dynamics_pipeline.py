from pathlib import Path

import pandas as pd

from app.data_engineering.pipelines.mouse_dynamics_pipeline import (
    run_mouse_dynamics_etl,
    stage_mouse_dynamics_raw_data,
)


def _write_session(path: Path, rows: list[tuple[float, float, str, str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["record timestamp,client timestamp,button,state,x,y"]
    for row in rows:
        lines.append(",".join([str(row[0]), str(row[1]), row[2], row[3], str(row[4]), str(row[5])]))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_stage_mouse_dynamics_raw_data_copies_expected_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "training_files" / "user1").mkdir(parents=True)
    (source_root / "test_files" / "user1").mkdir(parents=True)
    (source_root / "README.md").write_text("dataset docs", encoding="utf-8")
    (source_root / "public_labels.csv").write_text("filename,is_illegal\nsession_abc,1\n", encoding="utf-8")

    target_root = tmp_path / "target" / "mouse_dynamics_challenge"
    staged = stage_mouse_dynamics_raw_data(source_root, target_root)

    assert staged == target_root.resolve()
    assert (target_root / "training_files" / "user1").exists()
    assert (target_root / "test_files" / "user1").exists()
    assert (target_root / "README.md").exists()
    assert (target_root / "public_labels.csv").exists()


def test_run_mouse_dynamics_etl_builds_event_and_session_outputs(tmp_path: Path) -> None:
    raw_root = tmp_path / "mouse_raw"

    _write_session(
        raw_root / "training_files" / "user12" / "session_train_1",
        [
            (0.0, 0.0, "NoButton", "Move", 100, 100),
            (0.2, 0.2, "NoButton", "Move", 110, 104),
            (0.4, 0.4, "Left", "Pressed", 111, 104),
        ],
    )
    _write_session(
        raw_root / "test_files" / "user12" / "session_test_1",
        [
            (0.0, 0.0, "NoButton", "Move", 10, 10),
            (0.1, 0.1, "NoButton", "Drag", 20, 10),
            (0.2, 0.2, "Left", "Released", 22, 12),
        ],
    )
    (raw_root / "public_labels.csv").write_text(
        "filename,is_illegal\nsession_test_1,1\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "processed" / "mouse_events.csv"
    written_path = run_mouse_dynamics_etl(raw_root, output_path)

    assert written_path == output_path.resolve()
    assert output_path.exists()

    events = pd.read_csv(output_path)
    assert {"split", "user_id", "session_id", "distance", "speed", "is_illegal"}.issubset(events.columns)
    assert set(events["split"].unique()) == {"training", "test"}

    train_labels = events.loc[events["split"] == "training", "is_illegal"]
    test_labels = events.loc[events["split"] == "test", "is_illegal"]
    assert set(train_labels.unique()) == {0}
    assert set(test_labels.unique()) == {1}

    summary_path = output_path.with_name("mouse_events_session_summary.csv")
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert {"n_events", "duration", "mean_speed", "button_event_ratio"}.issubset(summary.columns)
    assert len(summary) == 2
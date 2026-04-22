import pandas as pd
import torch

from app.agents.biometric.mouse_authentication import (
    MouseAuthenticationModel,
    MouseAuthenticationModelConfig,
    MouseAuthenticationTrainer,
    MouseAuthenticationTrainingConfig,
    MouseSequenceFeatureConfig,
    compute_authentication_metrics,
    create_mouse_authentication_dataloaders,
    prepare_mouse_authentication_arrays,
)
from app.agents.biometric.mouse_authentication.train import _list_available_users


def _build_mouse_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_index in range(6):
        rows.extend(
            _build_session_rows(
                user_id="user_a",
                session_id=f"session_legit_{session_index}",
                split="training",
                base_x=10.0,
                base_y=20.0,
                delta_scale=0.2,
                dt=0.08,
                is_illegal=0,
            )
        )
        rows.extend(
            _build_session_rows(
                user_id="user_b",
                session_id=f"session_other_{session_index}",
                split="training",
                base_x=100.0,
                base_y=220.0,
                delta_scale=3.5,
                dt=0.45,
                is_illegal=0,
            )
        )
    return pd.DataFrame(rows)


def _build_session_rows(
    *,
    user_id: str,
    session_id: str,
    split: str,
    base_x: float,
    base_y: float,
    delta_scale: float,
    dt: float,
    is_illegal: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_position = base_x
    y_position = base_y
    for step in range(8):
        dx = delta_scale * (1.0 + step * 0.05)
        dy = delta_scale * 0.5
        x_position += dx
        y_position += dy
        distance = (dx**2 + dy**2) ** 0.5
        rows.append(
            {
                "split": split,
                "user_id": user_id,
                "session_id": session_id,
                "record_timestamp": float(step),
                "is_illegal": is_illegal,
                "dt": dt,
                "button_event": 1 if step % 4 == 0 else 0,
                "drag_event": 0 if user_id == "user_a" else 1,
                "pressed_event": 1 if step == 0 else 0,
                "released_event": 1 if step == 7 else 0,
                "x": x_position,
                "y": y_position,
                "dx": dx,
                "dy": dy,
                "distance": distance,
                "speed": distance / dt,
            }
        )
    return rows


def _build_mixed_split_mouse_events() -> pd.DataFrame:
    events = _build_mouse_events()
    extra_rows: list[dict[str, object]] = []
    extra_rows.extend(
        _build_session_rows(
            user_id="user_a",
            session_id="session_test_legit",
            split="test",
            base_x=15.0,
            base_y=30.0,
            delta_scale=0.3,
            dt=0.1,
            is_illegal=0,
        )
    )
    extra_rows.extend(
        _build_session_rows(
            user_id="user_b",
            session_id="session_test_illegal",
            split="test",
            base_x=120.0,
            base_y=240.0,
            delta_scale=3.2,
            dt=0.5,
            is_illegal=1,
        )
    )
    return pd.concat([events, pd.DataFrame(extra_rows)], ignore_index=True)


def test_prepare_mouse_authentication_arrays_builds_labeled_sequences() -> None:
    events = _build_mouse_events()
    feature_config = MouseSequenceFeatureConfig(max_sequence_length=8, min_sequence_length=4)

    timing_sequences, movement_sequences, lengths, labels = prepare_mouse_authentication_arrays(
        events,
        target_user_id="user_a",
        feature_config=feature_config,
    )

    assert timing_sequences.shape == (12, 8, len(feature_config.timing_features))
    assert movement_sequences.shape == (12, 8, len(feature_config.movement_features))
    assert lengths.tolist() == [8] * 12
    assert set(labels.tolist()) == {0.0, 1.0}
    assert int(labels.sum()) == 6


def test_prepare_mouse_authentication_arrays_respects_allowed_splits() -> None:
    events = _build_mixed_split_mouse_events()
    feature_config = MouseSequenceFeatureConfig(max_sequence_length=8, min_sequence_length=4)

    _, _, _, labels_all = prepare_mouse_authentication_arrays(
        events,
        target_user_id="user_a",
        feature_config=feature_config,
    )
    _, _, _, labels_training_only = prepare_mouse_authentication_arrays(
        events,
        target_user_id="user_a",
        feature_config=feature_config,
        allowed_splits=("training",),
    )

    assert len(labels_all) == 14
    assert int(labels_all.sum()) == 7
    assert len(labels_training_only) == 12
    assert int(labels_training_only.sum()) == 6


def test_list_available_users_respects_split_filter(tmp_path) -> None:
    events = pd.DataFrame(
        [
            {"split": "training", "user_id": "user_a"},
            {"split": "test", "user_id": "user_b"},
            {"split": "training", "user_id": "user_c"},
        ]
    )
    events_path = tmp_path / "events.csv"
    events.to_csv(events_path, index=False)

    users_training = _list_available_users(events_path, allowed_splits=("training",))
    users_all = _list_available_users(events_path, allowed_splits=None)

    assert users_training == ["user_a", "user_c"]
    assert users_all == ["user_a", "user_b", "user_c"]


def test_compute_authentication_metrics_returns_expected_values() -> None:
    metrics = compute_authentication_metrics(
        labels=torch.tensor([1, 1, 0, 0], dtype=torch.float32).numpy(),
        probabilities=torch.tensor([0.95, 0.87, 0.11, 0.08], dtype=torch.float32).numpy(),
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.eer == 0.0
    assert 0.08 <= metrics.eer_threshold <= 0.95


def test_mouse_authentication_trainer_runs_end_to_end() -> None:
    torch.manual_seed(7)
    events = _build_mouse_events()
    feature_config = MouseSequenceFeatureConfig(max_sequence_length=8, min_sequence_length=4)
    data_bundle = create_mouse_authentication_dataloaders(
        events,
        target_user_id="user_a",
        feature_config=feature_config,
        batch_size=4,
        validation_fraction=0.25,
        random_seed=11,
    )

    model = MouseAuthenticationModel(
        timing_input_dim=len(feature_config.timing_features),
        movement_input_dim=len(feature_config.movement_features),
        config=MouseAuthenticationModelConfig(
            lstm_hidden_size=8,
            lstm_layers=1,
            cnn_channels=(4, 8),
            projection_dim=8,
            dropout=0.1,
        ),
    )
    trainer = MouseAuthenticationTrainer(
        model,
        MouseAuthenticationTrainingConfig(
            batch_size=4,
            epochs=3,
            learning_rate=1e-3,
            decision_threshold=0.5,
            device="cpu",
        ),
    )

    history = trainer.fit(data_bundle.train_loader, data_bundle.validation_loader)
    evaluation = trainer.evaluate(data_bundle.validation_loader)
    probabilities = trainer.predict_probabilities(data_bundle.validation_loader)

    assert len(history["train"]) == 3
    assert len(history["validation"]) == 3
    assert probabilities.shape[0] == data_bundle.validation_size
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert 0.0 <= evaluation.precision <= 1.0
    assert 0.0 <= evaluation.recall <= 1.0
    assert 0.0 <= evaluation.f1 <= 1.0
    assert 0.0 <= evaluation.eer <= 1.0
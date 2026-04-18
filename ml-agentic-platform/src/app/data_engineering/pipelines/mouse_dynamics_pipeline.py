from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

_COLUMN_MAPPING = {
    "record timestamp": "record_timestamp",
    "client timestamp": "client_timestamp",
    "button": "button",
    "state": "state",
    "x": "x",
    "y": "y",
}


def stage_mouse_dynamics_raw_data(source_root: Path, target_root: Path) -> Path:
    """Copy the challenge dataset into the platform data/raw folder layout."""
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    required_dirs = ("training_files", "test_files")
    for dirname in required_dirs:
        src = source_root / dirname
        if not src.exists():
            raise FileNotFoundError(f"Expected source directory not found: {src}")
        shutil.copytree(src, target_root / dirname, dirs_exist_ok=True)

    for filename in ("public_labels.csv", "README.md"):
        src_file = source_root / filename
        if src_file.exists():
            shutil.copy2(src_file, target_root / filename)

    return target_root


def _load_session_frame(session_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(session_file)
    frame = frame.rename(columns=_COLUMN_MAPPING)
    missing = [col for col in _COLUMN_MAPPING.values() if col not in frame.columns]
    if missing:
        raise ValueError(f"Session file {session_file} is missing required columns: {missing}")

    frame = frame[list(_COLUMN_MAPPING.values())].copy()
    frame["record_timestamp"] = pd.to_numeric(frame["record_timestamp"], errors="coerce")
    frame["client_timestamp"] = pd.to_numeric(frame["client_timestamp"], errors="coerce")
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame = frame.dropna(subset=["record_timestamp", "client_timestamp", "x", "y"]).reset_index(drop=True)

    frame["dx"] = frame["x"].diff().fillna(0.0)
    frame["dy"] = frame["y"].diff().fillna(0.0)
    frame["dt"] = frame["record_timestamp"].diff().clip(lower=0.0).fillna(0.0)
    frame["distance"] = (frame["dx"] ** 2 + frame["dy"] ** 2) ** 0.5
    frame["speed"] = (frame["distance"] / frame["dt"].replace(0.0, pd.NA)).fillna(0.0)
    frame["button_event"] = (frame["button"].astype(str).str.lower() != "nobutton").astype(int)
    frame["drag_event"] = frame["state"].astype(str).str.lower().eq("drag").astype(int)
    frame["pressed_event"] = frame["state"].astype(str).str.lower().eq("pressed").astype(int)
    frame["released_event"] = frame["state"].astype(str).str.lower().eq("released").astype(int)
    return frame


def _iter_session_files(base_dir: Path, split: str) -> list[tuple[str, Path]]:
    split_dir = base_dir / f"{split}_files"
    if not split_dir.exists():
        return []

    session_files: list[tuple[str, Path]] = []
    for user_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        user_id = user_dir.name
        for session_file in sorted(user_dir.glob("session_*")):
            if session_file.is_file():
                session_files.append((user_id, session_file))
    return session_files


def _load_public_labels(labels_path: Path) -> dict[str, int]:
    if not labels_path.exists():
        return {}

    labels = pd.read_csv(labels_path)
    if "filename" not in labels.columns or "is_illegal" not in labels.columns:
        raise ValueError(f"Labels file {labels_path} must include filename and is_illegal columns")
    labels["filename"] = labels["filename"].astype(str)
    labels["is_illegal"] = pd.to_numeric(labels["is_illegal"], errors="coerce")
    labels = labels.dropna(subset=["is_illegal"])
    return dict(zip(labels["filename"], labels["is_illegal"].astype(int)))


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return path
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return path
    raise ValueError("output_path must end with .csv or .parquet")


def run_mouse_dynamics_etl(raw_path: Path, output_path: Path) -> Path:
    """Build event-level and session-level datasets from Balabit challenge files."""
    raw_path = raw_path.resolve()
    output_path = output_path.resolve()

    labels_map = _load_public_labels(raw_path / "public_labels.csv")
    event_frames: list[pd.DataFrame] = []

    for split in ("training", "test"):
        for user_id, session_file in _iter_session_files(raw_path, split):
            session_id = session_file.name
            frame = _load_session_frame(session_file)
            frame["split"] = split
            frame["user_id"] = user_id
            frame["session_id"] = session_id
            if split == "training":
                frame["is_illegal"] = 0
            else:
                frame["is_illegal"] = labels_map.get(session_id, pd.NA)
            event_frames.append(frame)

    if not event_frames:
        raise ValueError(f"No session files found under expected challenge layout in {raw_path}")

    events = pd.concat(event_frames, ignore_index=True)
    written_events_path = _write_frame(events, output_path)

    session_summary = (
        events.groupby(["split", "user_id", "session_id"], as_index=False)
        .agg(
            n_events=("session_id", "size"),
            duration=("record_timestamp", lambda s: float(s.max() - s.min())),
            total_distance=("distance", "sum"),
            mean_speed=("speed", "mean"),
            drag_ratio=("drag_event", "mean"),
            button_event_ratio=("button_event", "mean"),
            is_illegal=("is_illegal", "max"),
        )
    )
    summary_path = output_path.with_name(f"{output_path.stem}_session_summary{output_path.suffix}")
    _write_frame(session_summary, summary_path)

    return written_events_path

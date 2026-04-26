"""Shared pytest fixtures and helpers used across unit and integration tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Mouse event helpers
# ---------------------------------------------------------------------------

def build_session_rows(
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
    """Build a list of synthetic mouse event row dicts for one session."""
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
                "distance": distance,
                "speed": distance / dt if dt > 0 else 0.0,
            }
        )
    return rows


def build_mouse_events(num_sessions: int = 6) -> pd.DataFrame:
    """Return a DataFrame of synthetic mouse events for two users."""
    rows: list[dict[str, object]] = []
    for session_index in range(num_sessions):
        rows.extend(
            build_session_rows(
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
            build_session_rows(
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


# ---------------------------------------------------------------------------
# Mouse dynamics pipeline helper
# ---------------------------------------------------------------------------

def write_mouse_session(
    path: Path,
    rows: list[tuple[float, float, str, str, int, int]],
) -> None:
    """Write a minimal mouse session CSV file to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["record timestamp,client timestamp,button,state,x,y"]
    for row in rows:
        lines.append(",".join([str(row[0]), str(row[1]), row[2], row[3], str(row[4]), str(row[5])]))
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mouse_events_df() -> pd.DataFrame:
    """Pytest fixture that returns synthetic mouse event data."""
    return build_mouse_events()

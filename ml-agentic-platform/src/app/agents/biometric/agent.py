from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.biometric.mouse_authentication.inference import load_checkpoint, score_sequence

_logger = logging.getLogger(__name__)

# Default directory where the biometric training CLI writes per-user checkpoints.
_DEFAULT_ARTIFACTS_DIR = Path("artifacts/mouse_authentication")


class BiometricAgent:
    """Behavioral-biometrics scoring agent.

    In request-path inference the agent expects a ``biometric_payload`` dict with:

    * ``user_id`` (str) — resolves the per-user checkpoint file.
    * ``timing_sequence`` (list[list[float]]) — ``[T, 5]`` un-normalised timing features.
    * ``movement_sequence`` (list[list[float]]) — ``[T, 6]`` un-normalised movement features.
    * ``length`` (int, optional) — actual sequence length (defaults to ``len(timing_sequence)``).

    If ``timing_sequence`` / ``movement_sequence`` are absent, the agent falls back to a
    lightweight velocity-based heuristic so that existing callers remain unbroken.
    """

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else _DEFAULT_ARTIFACTS_DIR

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id: str | None = payload.get("user_id")
        timing_sequence: list | None = payload.get("timing_sequence")
        movement_sequence: list | None = payload.get("movement_sequence")

        if user_id and timing_sequence and movement_sequence:
            return self._score_with_model(
                user_id=user_id,
                timing_sequence=timing_sequence,
                movement_sequence=movement_sequence,
                length=payload.get("length"),
            )

        # Fallback: velocity heuristic for callers that supply no raw sequences
        _logger.debug("No timing/movement sequences in payload — using velocity heuristic fallback")
        velocity = float(payload.get("mean_velocity", 0.0))
        risk = max(0.0, min(1.0, 1.0 - velocity))
        return {
            "score": round(1.0 - risk, 4),
            "risk": round(risk, 4),
            "signal": "biometric_heuristic",
        }

    def _score_with_model(
        self,
        *,
        user_id: str,
        timing_sequence: list[list[float]],
        movement_sequence: list[list[float]],
        length: int | None,
    ) -> dict[str, Any]:
        checkpoint_path = self._artifacts_dir / f"{user_id}.pt"
        try:
            artifact = load_checkpoint(checkpoint_path)
        except FileNotFoundError:
            _logger.warning(
                "No checkpoint for user '%s' at %s — returning neutral score",
                user_id,
                checkpoint_path,
            )
            return {"score": 0.5, "risk": 0.5, "confidence": 0.0, "signal": "biometric_no_model"}

        result = score_sequence(
            artifact=artifact,
            timing_sequence=timing_sequence,
            movement_sequence=movement_sequence,
            length=length,
        )
        return {**result, "signal": "biometric_model", "user_id": user_id}

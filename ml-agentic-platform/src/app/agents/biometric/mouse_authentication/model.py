from __future__ import annotations

import torch
from torch import nn

from app.agents.biometric.mouse_authentication.config import MouseAuthenticationModelConfig


class MouseAuthenticationModel(nn.Module):
    def __init__(
        self,
        *,
        timing_input_dim: int,
        movement_input_dim: int,
        config: MouseAuthenticationModelConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or MouseAuthenticationModelConfig()
        if not config.cnn_channels:
            raise ValueError("cnn_channels must contain at least one output channel")

        self._lstm = nn.LSTM(
            input_size=timing_input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        conv_layers: list[nn.Module] = []
        in_channels = movement_input_dim
        for out_channels in config.cnn_channels:
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self._movement_encoder = nn.Sequential(*conv_layers)

        lstm_output_dim = config.lstm_hidden_size * 2
        cnn_output_dim = config.cnn_channels[-1]
        self._classifier = nn.Sequential(
            nn.Linear(lstm_output_dim + cnn_output_dim, config.projection_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim, 1),
        )

    def forward(
        self,
        timing_sequence: torch.Tensor,
        movement_sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            timing_sequence,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden_state, _) = self._lstm(packed_sequence)
        forward_hidden = hidden_state[-2]
        backward_hidden = hidden_state[-1]
        timing_embedding = torch.cat([forward_hidden, backward_hidden], dim=1)

        movement_features = movement_sequence.transpose(1, 2)
        movement_embedding = self._movement_encoder(movement_features)
        movement_embedding = self._masked_temporal_mean(movement_embedding, lengths)

        fused_representation = torch.cat([timing_embedding, movement_embedding], dim=1)
        return self._classifier(fused_representation).squeeze(-1)

    @staticmethod
    def _masked_temporal_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        time_steps = values.shape[-1]
        mask = torch.arange(time_steps, device=values.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(1)
        masked_values = values * mask
        normalizer = lengths.clamp(min=1).to(values.dtype).unsqueeze(1)
        return masked_values.sum(dim=-1) / normalizer
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import scatter

from app.agents.graph_fraud.temporal_gnn.config import TemporalModelConfig


class TimeEncoder(nn.Module):
    """Sinusoidal time encoder similar to TGAT-style periodic features."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        self.output_dim = output_dim

        half_dim = max(1, output_dim // 2)
        frequency = torch.arange(half_dim, dtype=torch.float32)
        frequency = torch.exp(-math.log(10000.0) * frequency / max(half_dim - 1, 1))
        self.register_buffer("frequency", frequency)

    def forward(self, edge_time: torch.Tensor) -> torch.Tensor:
        t = edge_time.unsqueeze(-1) * self.frequency
        encoded = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        if encoded.shape[-1] < self.output_dim:
            pad_width = self.output_dim - encoded.shape[-1]
            encoded = F.pad(encoded, (0, pad_width))
        return encoded[:, : self.output_dim]


class TemporalGraphModel(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_attr_dim: int,
        config: TemporalModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TemporalModelConfig()

        if self.config.backbone not in {"gcn", "gat"}:
            raise ValueError("backbone must be one of: gcn, gat")

        self.time_encoder = TimeEncoder(self.config.time_encoding_dim)
        combined_input_dim = node_feature_dim + edge_attr_dim + self.config.time_encoding_dim
        self.input_projection = nn.Linear(combined_input_dim, self.config.hidden_dim)

        self.convs = nn.ModuleList()
        if self.config.backbone == "gcn":
            for _ in range(self.config.num_layers):
                self.convs.append(GCNConv(self.config.hidden_dim, self.config.hidden_dim))
        else:
            if self.config.hidden_dim % self.config.gat_heads != 0:
                raise ValueError("hidden_dim must be divisible by gat_heads for GAT")
            out_per_head = self.config.hidden_dim // self.config.gat_heads
            for _ in range(self.config.num_layers):
                self.convs.append(
                    GATConv(
                        self.config.hidden_dim,
                        out_per_head,
                        heads=self.config.gat_heads,
                        concat=True,
                        dropout=self.config.dropout,
                    )
                )

        self.classifier = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_time = data.edge_time
        edge_attr = data.edge_attr

        temporal_edge_embedding = self.time_encoder(edge_time)
        destination_nodes = edge_index[1]

        node_temporal_context = scatter(
            temporal_edge_embedding,
            destination_nodes,
            dim=0,
            dim_size=x.shape[0],
            reduce="mean",
        )
        node_edge_context = scatter(
            edge_attr,
            destination_nodes,
            dim=0,
            dim_size=x.shape[0],
            reduce="mean",
        )

        x = torch.cat([x, node_temporal_context, node_edge_context], dim=-1)
        x = F.relu(self.input_projection(x))

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)

        logits = self.classifier(x).squeeze(-1)
        return logits

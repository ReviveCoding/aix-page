"""Compact DCNv2-style crossed embedding model for aggregated KDD rows."""

from __future__ import annotations

import torch
from torch import nn


class CrossLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Linear(width, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.zeros_(self.weight.weight)

    def forward(self, base: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        output = base * self.weight(current) + self.bias + current
        if not isinstance(output, torch.Tensor):
            raise TypeError("cross layer must produce a Tensor")
        return output


class DCNv2(nn.Module):
    def __init__(
        self,
        categorical_fields: int,
        buckets: int,
        embedding_dim: int,
        dense_features: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(buckets, embedding_dim) for _ in range(categorical_fields)]
        )
        for embedding in self.embeddings:
            if not isinstance(embedding, nn.Embedding):
                raise TypeError("categorical embedding contract requires nn.Embedding")
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        width = categorical_fields * embedding_dim + dense_features
        self.crosses = nn.ModuleList([CrossLayer(width), CrossLayer(width)])
        self.deep = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Linear(width + 64, 1)

    def forward(self, categorical: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)]
        base = torch.cat([*embedded, dense], dim=1)
        crossed = base
        for layer in self.crosses:
            crossed = layer(base, crossed)
        output = self.output(torch.cat([crossed, self.deep(base)], dim=1)).squeeze(1)
        if not isinstance(output, torch.Tensor):
            raise TypeError("DCNv2 must produce a Tensor")
        return output

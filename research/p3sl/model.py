"""Reference CNN architecture used throughout the P3SL experiments.

The architecture is deliberately identical to the model defined in
``my_implementation/P3SL_Implementation/1st_Demo`` so that results from the
Docker simulation and the in-process research harness are directly comparable.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Architecture definition
# ---------------------------------------------------------------------------
# Index → Layer
#   0 Conv2d(C_in, 32)            — first feature extractor
#   1 GroupNorm(8, 32)
#   2 ReLU
#   3 Conv2d(32, 32)
#   4 GroupNorm(8, 32)
#   5 ReLU
#   6 MaxPool2d(2)
#   7 Conv2d(32, 64)
#   8 GroupNorm(8, 64)
#   9 ReLU
#  10 Conv2d(64, 64)
#  11 GroupNorm(8, 64)
#  12 ReLU
#  13 MaxPool2d(2)
#  14 AdaptiveAvgPool2d(1)
#  15 Flatten
#  16 Dropout(0.2)
#  17 Linear(64, num_classes)
# ---------------------------------------------------------------------------

#: Maximum *valid* split layer index. Splits beyond layer 13 leave only fully
#: connected / pooling stubs on the client which is degenerate for a CNN study.
MAX_SPLIT_LAYER: int = 13

#: Recommended set of split layers tracked in the paper.
DEFAULT_SPLIT_POINTS: Tuple[int, ...] = (2, 5, 6, 9, 10, 13)


class P3SLModel(nn.Module):
    """CNN classifier that supports arbitrary front/back splits."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            ]
        )

    # ------------------------------------------------------------------ utils
    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward_upto(self, x: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Run the *client* (head) computation through ``layers[0 .. split_layer]``."""
        if not 0 <= split_layer < self.num_layers:
            raise ValueError(
                f"split_layer={split_layer} out of range [0, {self.num_layers - 1}]"
            )
        for i in range(0, split_layer + 1):
            x = self.layers[i](x)
        return x

    def forward_from(self, x: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Run the *server* (tail) computation through ``layers[split_layer+1 .. end]``."""
        if not 0 <= split_layer < self.num_layers:
            raise ValueError(
                f"split_layer={split_layer} out of range [0, {self.num_layers - 1}]"
            )
        for i in range(split_layer + 1, self.num_layers):
            x = self.layers[i](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = layer(x)
        return x

    # ------------------------------------------------------ parameter views
    def head_parameters(self, split_layer: int) -> List[nn.Parameter]:
        """Parameters of layers ``0 .. split_layer`` (the client-owned head)."""
        params: List[nn.Parameter] = []
        for i in range(0, split_layer + 1):
            params.extend(self.layers[i].parameters())
        return params

    def tail_parameters(self, split_layer: int) -> List[nn.Parameter]:
        """Parameters of layers ``split_layer+1 .. end`` (the server-owned tail)."""
        params: List[nn.Parameter] = []
        for i in range(split_layer + 1, self.num_layers):
            params.extend(self.layers[i].parameters())
        return params


def ir_shape_for_split(model: P3SLModel, split_layer: int, dummy_input: torch.Tensor) -> torch.Size:
    """Probe-infer the shape of the IR produced at a given split layer."""
    with torch.no_grad():
        out = model.forward_upto(dummy_input, split_layer)
    return out.shape

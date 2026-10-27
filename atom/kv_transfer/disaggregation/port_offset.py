# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Worker port offsets and PP region mappings for KV disaggregation."""

from __future__ import annotations

from collections.abc import Sequence


def layer_region_indices(
    groups: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[list[int], int]:
    """Map field-major regions from local to global layer IDs.

    Each group supplies ``(local_layer_ids, global_layer_ids)`` in allocation
    order. Fields may cover different subsets of layers (e.g. an index cache).
    Return the explicit consumer indices and the full consumer region count,
    which is also the offset at which an independent draft pool starts.
    """
    indices, offset = [], 0
    for local_layers, global_layers in groups:
        slots = {layer: offset + i for i, layer in enumerate(global_layers)}
        if len(slots) != len(global_layers):
            raise ValueError("Global layer IDs must be unique within a region group")
        if any(layer not in slots for layer in local_layers):
            raise ValueError("Local region layers are missing from the global layout")
        indices.extend(slots[layer] for layer in local_layers)
        offset += len(global_layers)
    return indices, offset


def side_channel_port_offset(
    dp_rank: int,
    tp_rank: int,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    dp_size: int = 1,
) -> int:
    """Return the unique port offset for a worker's (pp, dp, tp) position."""
    return pp_rank * (dp_size * tp_size) + dp_rank * tp_size + tp_rank


def consumer_region_indices(
    num_local_regions: int,
    num_local_layers: int,
    start_layer: int,
    num_consumer_regions: int,
    pp_size: int,
) -> list[int] | None:
    """Map a PP stage's local RDMA regions to consumer indices (group-major).

    Stride is derived from ``num_consumer_regions``, not ``num_hidden_layers``,
    so spec-decode's extra draft layer is handled automatically.

    Returns None when the layout is not uniform group-major.
    """
    if pp_size == 1 or num_local_layers == 0 or num_local_regions == 0:
        return list(range(num_local_regions))
    groups, remainder = divmod(num_local_regions, num_local_layers)
    if remainder != 0:
        return None
    stride, extra = divmod(num_consumer_regions, groups)
    if extra != 0:
        return None
    indices = [
        g * stride + start_layer + layer
        for g in range(groups)
        for layer in range(num_local_layers)
    ]
    if indices[-1] >= num_consumer_regions:
        return None
    return indices

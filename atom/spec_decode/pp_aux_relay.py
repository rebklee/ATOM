# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Carry a drafter's declared target taps across pipeline stages.

The drafter lives on the last PP stage. Its hooks for earlier layers sit on
placeholders and never fire. Each earlier stage captures the layers it runs
and sends those tensors with its hidden states; the last stage hands them to
the drafter through the same buffers used by local capture.
"""

import torch

from atom.distributed.pp_comm import PP_DRAFT_AUX_PREFIX
from atom.models.utils import PPMissingLayer
from atom.spec_decode.drafter import AuxCaptureSpec, Drafter


class PPAuxRelay:
    def __init__(self, config, target_model, drafter, device, spec: AuxCaptureSpec):
        self.layer_ids = spec.layer_ids
        held = {
            lid: (module, extract)
            for lid, module, extract in spec.resolve_taps(target_model)
            if not isinstance(module, PPMissingLayer)
        }
        # The last stage taps its own layers through the drafter's hooks and
        # must be handed every other one.
        self._expected = set(self.layer_ids) - set(held) if drafter else set()
        self._received: dict[int, torch.Tensor] = {}
        self._captures: dict[int, torch.Tensor] = {}
        for i in held if drafter is None else []:
            self._captures[i] = torch.zeros(
                config.max_num_batched_tokens,
                spec.hidden_size,
                dtype=config.torch_dtype,
                device=device,
            )
            module, extract = held[i]
            Drafter.register_aux_capture(
                module,
                self._captures[i],
                extract,
                "output" if i == -1 else spec.capture,
            )

    def take_received(self, payload) -> None:
        """Pop the earlier stages' taps off a received payload."""
        self._received = {
            i: payload.tensors.pop(f"{PP_DRAFT_AUX_PREFIX}{i}")
            for i in self.layer_ids
            if f"{PP_DRAFT_AUX_PREFIX}{i}" in payload.tensors
        }

    def attach(self, payload) -> None:
        """Send on every tap held so far, received and this stage's own."""
        num_tokens = payload.tensors["hidden_states"].shape[0]
        taps = {i: buf[:num_tokens] for i, buf in self._captures.items()}
        for i, tensor in {**self._received, **taps}.items():
            payload.tensors[f"{PP_DRAFT_AUX_PREFIX}{i}"] = tensor

    def deliver(self, drafter) -> None:
        """On the last stage, hand the drafter the taps it cannot capture."""
        if self._received.keys() != self._expected:
            raise RuntimeError(
                f"expected taps {sorted(self._expected)}, got {sorted(self._received)}"
            )
        for i, tensor in self._received.items():
            drafter.accept_relayed_aux(i, tensor)

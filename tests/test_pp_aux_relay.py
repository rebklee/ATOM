# SPDX-License-Identifier: MIT
"""Declared drafter taps, carried across pipeline stages.

Three stages over real hooks and a real payload dict; the last stage must
receive exactly what one stage holding the whole stack would have captured.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from atom.models.utils import IntermediateTensors, PPMissingLayer

TAPS, HIDDEN = [1, 3, 5], 8
CONFIG = SimpleNamespace(
    speculative_config=SimpleNamespace(
        use_dspark=lambda: True,
        draft_model_hf_config=SimpleNamespace(dspark_target_layer_ids=TAPS),
    ),
    max_num_batched_tokens=16,
    hf_config=SimpleNamespace(hidden_size=HIDDEN),
    torch_dtype=torch.float32,
)


def _spec(capture="output", embedding=False):
    from atom.spec_decode.drafter import AuxCaptureSpec

    return AuxCaptureSpec(
        tuple(([-1] if embedding else []) + TAPS),
        HIDDEN,
        lambda value, _: value[0] if isinstance(value, tuple) else value,
        capture=capture,
    )


class _Layer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x * 1.5 + self.idx


class _Stage(nn.Module):
    def __init__(self, held):
        super().__init__()
        self.embed_tokens = nn.Identity() if 0 in held else PPMissingLayer()
        self.layers = nn.ModuleList(
            _Layer(i) if i in held else PPMissingLayer() for i in range(6)
        )

    def run(self, x):
        if not isinstance(self.embed_tokens, PPMissingLayer):
            x = self.embed_tokens(x)
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                x = layer(x)
        return x


class _Drafter:
    model = None

    def __init__(self):
        self.got = {}

    def accept_relayed_aux(self, layer_id, tensor):
        self.got[layer_id] = tensor.clone()


@pytest.fixture(scope="module")
def capture_modules():
    # Run real capture/proposer/factory code with CPU tensors. Weight loading
    # and token emission are outside this contract and import unrelated native
    # AITER kernels. Refuse either if a test accidentally reaches it.
    def unused(*args, **kwargs):
        raise AssertionError("Aux capture must not load weights or emit token IDs")

    loader = ModuleType("atom.model_loader.loader")
    loader.load_model = unused
    embed = ModuleType("atom.model_ops.embed_head")
    embed.empty_token_ids = unused
    with patch.dict(sys.modules, {loader.__name__: loader, embed.__name__: embed}):
        drafter = importlib.import_module("atom.spec_decode.drafter")
        relay = importlib.import_module("atom.spec_decode.pp_aux_relay")
        yield SimpleNamespace(drafter=drafter, relay=relay)


@pytest.fixture
def relay_mod(capture_modules):
    return capture_modules.relay


@pytest.fixture(autouse=True)
def _target_forward(monkeypatch, capture_modules):
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_draft=False, ubatch_token_offset=0)
    )
    monkeypatch.setattr(capture_modules.drafter, "get_forward_context", lambda: ctx)


def _send(stage, relay, h):
    h = stage.run(h)
    payload = IntermediateTensors({"hidden_states": h, "residual": h})
    relay.attach(payload)
    return payload


def _capture_drafter(spec, target):
    from atom.spec_decode.dspark_proposer import DSparkProposer

    drafter = DSparkProposer.__new__(DSparkProposer)
    drafter.max_num_tokens = CONFIG.max_num_batched_tokens
    drafter.device, drafter.dtype = "cpu", CONFIG.torch_dtype
    drafter.config = SimpleNamespace(
        hf_config=CONFIG.hf_config,
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(
                dspark_target_layer_ids=spec.layer_ids
            )
        ),
    )
    drafter.model = SimpleNamespace()
    if spec.capture == "input":
        drafter.model.target_aux_capture_spec = lambda ids, hidden: type(spec)(
            ids, hidden, spec.extract, capture="input"
        )
    drafter.arm_aux_capture(target)
    return drafter


@pytest.mark.parametrize("capture", ["input", "output"])
@pytest.mark.parametrize("embedding", [False, True])
def test_last_stage_receives_every_tap_it_does_not_run(relay_mod, capture, embedding):
    x = torch.randn(5, HIDDEN)
    expect, h = {-1: x.clone()}, x
    for i in range(6):
        if capture == "input":
            expect[i] = h.clone()
        h = h * 1.5 + i
        if capture == "output":
            expect[i] = h.clone()
    stage0, stage1, stage2 = _Stage({0, 1}), _Stage({2, 3}), _Stage({4, 5})
    spec = _spec(capture, embedding)
    drafter = _capture_drafter(spec, stage2)
    relay0 = relay_mod.PPAuxRelay(CONFIG, stage0, None, "cpu", spec)
    relay1 = relay_mod.PPAuxRelay(CONFIG, stage1, None, "cpu", spec)
    relay2 = relay_mod.PPAuxRelay(CONFIG, stage2, drafter, "cpu", spec)

    payload = _send(stage0, relay0, x)
    relay1.take_received(payload)
    assert set(payload.tensors) == {"hidden_states", "residual"}
    payload = _send(stage1, relay1, payload["hidden_states"])
    relay2.take_received(payload)
    output = stage2.run(payload["hidden_states"])
    relay2.deliver(drafter)
    for i, tensor in zip(spec.layer_ids, drafter.aux_for(output), strict=True):
        torch.testing.assert_close(tensor, expect[i])


def test_last_stage_refuses_a_forward_missing_a_tap(relay_mod):
    relay = relay_mod.PPAuxRelay(CONFIG, _Stage({4, 5}), _Drafter(), "cpu", _spec())
    relay.take_received(
        IntermediateTensors(
            {"hidden_states": torch.zeros(2, 8), "draft_aux_1": torch.zeros(2, 8)}
        )
    )
    with pytest.raises(RuntimeError, match=r"expected taps \[1, 3\]"):
        relay.deliver(_Drafter())


def test_mtp_has_no_cross_stage_aux_contract():
    from atom.spec_decode.factory import build_pp_aux_relay

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(use_dspark=lambda: False)
    )
    assert build_pp_aux_relay(config, _Stage({0, 1}), None, "cpu") is None


def test_pp_uses_the_same_dspark_contract_as_local_capture():
    from atom.spec_decode.dspark_proposer import DSparkProposer
    from atom.spec_decode.factory import build_pp_aux_relay

    spec = DSparkProposer.config_aux_capture_spec(CONFIG)
    drafter = SimpleNamespace(_aux_capture_spec=lambda _: spec)
    relay = build_pp_aux_relay(CONFIG, _Stage({4, 5}), drafter, "cpu")
    assert relay.layer_ids == tuple(TAPS)
    drafter._aux_capture_spec = lambda _: _spec("input")
    with pytest.raises(NotImplementedError, match="configuration contract"):
        build_pp_aux_relay(CONFIG, _Stage({4, 5}), drafter, "cpu")


@pytest.mark.parametrize("layer_ids", [(1, 1), (3, 1), (6,), (-2,)])
def test_pp_and_local_capture_reject_the_same_invalid_taps(
    relay_mod, capture_modules, layer_ids
):
    spec = capture_modules.drafter.AuxCaptureSpec(
        layer_ids, HIDDEN, lambda value, _: value
    )
    with pytest.raises(ValueError):
        relay_mod.PPAuxRelay(CONFIG, _Stage({0, 1}), None, "cpu", spec)


def test_pp_payload_keeps_relayed_taps_and_drops_strangers():
    pp_comm = pytest.importorskip("atom.distributed.pp_comm")
    payload = IntermediateTensors(
        {
            "hidden_states": 1,
            "residual": 2,
            "draft_aux_17": 3,
            "other": 4,
            "sparse_kv_indices": 5,
        }
    )
    assert set(pp_comm._stage_payload(payload)) == {
        "hidden_states",
        "residual",
        "draft_aux_17",
        "sparse_kv_indices",
    }

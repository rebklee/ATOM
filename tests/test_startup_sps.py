# SPDX-License-Identifier: MIT
"""DSpark's startup SPS sweep: table shape, cache reuse, and rank agreement.

The GPU replay itself is faked; what is checked is everything around it --
which shapes are timed, that every rank takes the same cached/measured branch,
and that a restart of the same deployment reuses the measurements.
"""

import itertools
from types import SimpleNamespace as NS

import pytest
import torch

from atom.spec_decode import startup_sps
from atom.spec_decode.startup_sps import sps_table_from_points


def test_table_is_a_monotone_latency_envelope_over_rows():
    table = sps_table_from_points([(2, 10.0), (4, 9.0), (6, 14.0)], 6)
    latency = [1000 / v for v in table]
    assert latency[2] == pytest.approx(10.0)
    assert latency[4] == pytest.approx(10.0), "a faster larger graph is not trusted"
    assert latency[6] == pytest.approx(14.0)
    assert all(b >= a for a, b in itertools.pairwise(latency))


def test_uncovered_rows_extend_along_the_last_slope_only_when_asked():
    points = [(6, 10.0), (12, 13.0)]
    with pytest.raises(ValueError, match="cover"):
        sps_table_from_points(points, 18)
    latency = [1000 / v for v in sps_table_from_points(points, 18, extrapolate=True)]
    assert latency[18] == pytest.approx(16.0)


def _runner(tmp_path, monkeypatch, *, cached_elsewhere=True):
    monkeypatch.setenv("ATOM_DSPARK_SPS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "atom.spec_decode.calibration.calibration_identity",
        lambda config, device: {"deployment": "x"},
    )
    monkeypatch.setattr(
        "atom.utils.region_graph.TokenGraph.captured_sizes",
        staticmethod(lambda phase=None: (2, 4, 6, 8, 12, 24)),
    )
    # Two ranks: the peer reports whether ITS cache read succeeded.
    monkeypatch.setattr(
        startup_sps, "_all_true", lambda flag: flag and cached_elsewhere
    )
    monkeypatch.setattr(startup_sps, "_all_max", lambda value: value)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a: 0)
    return NS(config=NS(max_num_seqs=4), device="cpu", rank=0, capture_sizes=[4, 1, 2])


def test_a_restart_of_the_same_deployment_reuses_the_sweep(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    timed = []

    def measure(runner, shapes):
        timed.append(list(shapes))
        return [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 5.0 + b * q}
            for b, q in shapes
        ]

    monkeypatch.setattr(startup_sps, "_measure", measure)
    first = startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    # Only captured (bs, q) products, batches ascending.
    assert timed == [[(1, 6), (2, 1), (2, 3), (2, 6), (4, 1), (4, 3), (4, 6)]]
    assert len(list(tmp_path.glob("*.json"))) == 1
    second = startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert len(timed) == 1, "the cached curve must be reused"
    assert torch.equal(first, second)
    assert first.numel() == 25


def test_one_rank_missing_the_cache_makes_every_rank_measure(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 1.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    runner = _runner(tmp_path, monkeypatch, cached_elsewhere=False)
    measured = []
    monkeypatch.setattr(
        startup_sps,
        "_measure",
        lambda runner, shapes: measured.append(1)
        or [
            {"batch": b, "q": q, "rows": b * q, "target_ms": 1.0 + b * q}
            for b, q in shapes
        ],
    )
    startup_sps.startup_sps_table(runner, [1, 3, 6], 24)
    assert measured == [1], "a local hit must not skip a sweep its peers run"


def test_no_token_graph_to_time_is_refused(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "atom.utils.region_graph.TokenGraph.captured_sizes",
        staticmethod(lambda phase=None: ()),
    )
    with pytest.raises(RuntimeError, match="TR_HYBRID"):
        startup_sps.startup_sps_table(runner, [1, 3, 6], 24)

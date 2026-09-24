"""
Self-tests: prove the equivalence check passes a no-op and catches both ways an
optimizer can break a kernel. These run regardless of --atlas-opt.
"""

import re

import pytest

from tests import harness

KERNELS = {k.name: k for k in harness.discover_kernels()}
MATMUL = KERNELS["ParameterizedMatmul32x32x32Program"]


def _check(kernel, optimizer, hardware_config_cls, tmp_path):
    return harness.check_equivalence(
        kernel, optimizer, hardware_config_cls(), workdir=tmp_path, max_cycles=100000
    )


def test_optimizer_receives_original_source() -> None:
    assert MATMUL.asm_path is not None
    assert MATMUL.asm_path.name == "parameterized_matmul32x32x32.S"
    assert "loop_1:" in MATMUL.source()


def test_identity_is_equivalent(hardware_config_cls, tmp_path) -> None:
    before, after = _check(MATMUL, harness.identity_optimizer, hardware_config_cls, tmp_path)
    assert before.cycles == after.cycles
    assert {"dram_output", "vmem"} <= set(before.regions)
    assert any(label.startswith("dram_input@") for label in before.regions)


def test_missing_delays_are_caught(hardware_config_cls, tmp_path) -> None:
    with pytest.raises(harness.EquivalenceError, match="raised during simulation"):
        _check(MATMUL, harness.strip_delays_optimizer, hardware_config_cls, tmp_path)


def test_silently_wrong_output_is_caught(hardware_config_cls, tmp_path) -> None:
    """A kernel that runs cleanly but never writes its result must still fail."""

    def drop_stores(source, workdir):
        return re.sub(r"^\s*dma\.store\.ch\d.*$", "", source, flags=re.MULTILINE)

    with pytest.raises(harness.EquivalenceError, match=r"dram_output: \d+/2048 bytes differ"):
        _check(MATMUL, drop_stores, hardware_config_cls, tmp_path)

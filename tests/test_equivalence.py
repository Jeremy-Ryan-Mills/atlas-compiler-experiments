"""
For every kernel registered in npu_model: run it, optimize it, run it again, and
require the optimized kernel to leave the same architectural state behind.

The optimizer comes from --atlas-opt / $ATLAS_OPT (see conftest.py).
"""

import pytest

from tests import harness

KERNELS = harness.discover_kernels()


def test_kernel_registry_is_not_empty() -> None:
    assert KERNELS, "npu_model.configs.programs registered no kernels"


@pytest.mark.parametrize("kernel", KERNELS, ids=[k.name for k in KERNELS])
def test_optimized_kernel_is_equivalent(
    kernel: harness.Kernel,
    optimizer: harness.Optimizer,
    hardware_config_cls,
    pytestconfig: pytest.Config,
    workdir,
    record_cycles,
) -> None:
    try:
        before, after = harness.check_equivalence(
            kernel,
            optimizer,
            hardware_config_cls(),
            workdir=workdir,
            max_cycles=pytestconfig.getoption("max_cycles"),
            live_state=pytestconfig.getoption("live_state"),
        )
    except harness.BaselineError as exc:
        pytest.fail(f"BASELINE (npu_model problem, not the optimizer): {exc}")
    except harness.OptimizerError as exc:
        pytest.fail(f"optimizer failed on {kernel.name}:\n{exc}")
    except harness.EquivalenceError as exc:
        pytest.fail(f"optimized {kernel.name} {exc}")
    record_cycles(kernel.name, before.cycles, after.cycles)

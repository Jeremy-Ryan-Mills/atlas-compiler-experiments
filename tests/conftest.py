import os
import shlex
import shutil
from pathlib import Path

import pytest

from tests import harness

# Must happen before any test module imports npu_model's program registry.
harness.bootstrap_npu_model()

DEFAULT_OPTIMIZER = harness.REPO_ROOT / "build" / "atlas-opt"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("atlas equivalence")
    group.addoption(
        "--atlas-opt",
        default=os.environ.get("ATLAS_OPT"),
        help=(
            "Optimizer to test: a path/command invoked as `<cmd> in.S -o out.S`, or a "
            f"builtin ({', '.join(harness.BUILTIN_OPTIMIZERS)}). Defaults to $ATLAS_OPT, "
            f"then {DEFAULT_OPTIMIZER.relative_to(harness.REPO_ROOT)} if it exists."
        ),
    )
    group.addoption(
        "--atlas-opt-args",
        default=os.environ.get("ATLAS_OPT_ARGS", ""),
        help="Extra flags passed to an external optimizer, e.g. '-O3'.",
    )
    group.addoption(
        "--max-cycles",
        type=int,
        default=100000,
        help="Cycle budget per run (a kernel's own `kernel_max_cycles` wins).",
    )
    group.addoption(
        "--hardware-config",
        default="DefaultHardwareConfig",
        help="npu_model hardware configuration class.",
    )
    group.addoption(
        "--no-compare-vmem",
        action="store_true",
        help="Only compare DRAM, for optimizers allowed to change VMEM scratch contents.",
    )
    group.addoption(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Keep each kernel's before.S/after.S here instead of a temp dir.",
    )


@pytest.fixture(scope="session")
def optimizer(pytestconfig: pytest.Config) -> harness.Optimizer:
    choice = pytestconfig.getoption("atlas_opt")
    if choice is None and DEFAULT_OPTIMIZER.exists():
        choice = str(DEFAULT_OPTIMIZER)
    if choice is None:
        pytest.skip(
            "no optimizer: build atlas-opt to build/atlas-opt, or pass "
            "--atlas-opt=<path|identity|strip-delays>"
        )
    if choice in harness.BUILTIN_OPTIMIZERS:
        return harness.BUILTIN_OPTIMIZERS[choice]

    command = shlex.split(choice)
    if shutil.which(command[0]) is None:
        raise pytest.UsageError(f"--atlas-opt: `{command[0]}` is not an executable")
    command += shlex.split(pytestconfig.getoption("atlas_opt_args"))
    return harness.external_optimizer(command)


@pytest.fixture(scope="session")
def hardware_config_cls(pytestconfig: pytest.Config):
    import npu_model.configs.hardware as hardware_configs

    name = pytestconfig.getoption("hardware_config")
    try:
        return getattr(hardware_configs, name)
    except AttributeError as exc:
        raise pytest.UsageError(f"Unknown hardware config '{name}'") from exc


@pytest.fixture
def workdir(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    root = request.config.getoption("artifacts_dir")
    if root is None:
        return tmp_path
    path = root / request.node.callspec.id
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Cycle reporting
# ---------------------------------------------------------------------------

_cycles_key = pytest.StashKey[dict[str, tuple[int, int]]]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_cycles_key] = {}


@pytest.fixture
def record_cycles(request: pytest.FixtureRequest):
    def record(kernel: str, before: int, after: int) -> None:
        request.config.stash[_cycles_key][kernel] = (before, after)
        request.node.user_properties.append(("cycles", (before, after)))

    return record


def _describe(before: int, after: int) -> str:
    saved = before - after
    return (
        f"{before:>8} -> {after:>8} cycles  saved {saved:>7} "
        f"({saved / before:6.1%})  {before / after:5.2f}x"
    )


def _cycles_of(report: pytest.TestReport) -> tuple[int, int] | None:
    if report.when != "call" or not report.passed:
        return None
    return dict(report.user_properties).get("cycles")


def pytest_report_teststatus(report: pytest.TestReport, config: pytest.Config):
    """Kernel tests show their cycle change instead of a bare `.` / `PASSED`."""
    cycles = _cycles_of(report)
    if cycles is None:
        return None
    return "passed", "", ("PASSED  " + _describe(*cycles), {"green": True})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Without -v there is no per-test line to extend, so print one."""
    cycles = _cycles_of(report)
    reporter = _terminal_reporter
    if cycles is None or reporter is None or reporter.verbosity > 0:
        return
    # Write through the terminal writer: `reporter.write_line` resets pytest's
    # line state, which makes it re-print the file name before the next test.
    writer = reporter._tw
    if writer.width_of_current_line:  # finish a pending line of `.`s
        writer.line()
    name = report.nodeid.rsplit("[", 1)[-1].rstrip("]")
    writer.line(f"{name + ':':<{_name_width + 1}} {_describe(*cycles)}")


_terminal_reporter = None
_name_width = 0


def pytest_collection_finish(session: pytest.Session) -> None:
    global _name_width
    ids = [item.callspec.id for item in session.items if hasattr(item, "callspec")]
    _name_width = max(map(len, ids), default=0)


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    global _terminal_reporter
    _terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    cycles = config.stash.get(_cycles_key, {})
    if not cycles:
        return
    terminalreporter.section("cycles before -> after optimization")
    width = max(len(name) for name in [*cycles, "TOTAL"])
    for name, (before, after) in sorted(cycles.items()):
        terminalreporter.write_line(f"{name:<{width}}  {_describe(before, after)}")
    total_before = sum(before for before, _ in cycles.values())
    total_after = sum(after for _, after in cycles.values())
    terminalreporter.write_line(f"{'TOTAL':<{width}}  {_describe(total_before, total_after)}")
    terminalreporter.write_line(
        f"({len(cycles)} equivalent kernels; failed kernels are not counted)"
    )

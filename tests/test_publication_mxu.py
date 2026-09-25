"""Check MXU results and source lifetime at first publication."""

import contextlib
from dataclasses import replace
import io
import json

import pytest
import torch

from tests import harness
from tests.test_publication import PUBLISH, publication_compiler, publication_dir


def observe_mxu(source, hardware_config_cls, directory, unit, operation):
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'program.S').write_text(source)
    config = hardware_config_cls()
    config.arch_state_config = replace(config.arch_state_config, dram_size=8192)
    sim = Simulation(
        hardware_config=config,
        logger_config=LoggerConfig(filename=str(directory / 'trace.json')),
        program=harness.program_from_asm(source, []),
        verbose=False,
    )
    try:
        core = sim.core
        core.reset()
        state = core.arch_state
        state.read_mrf_fp8(0).fill_(1)
        state.read_mrf_fp8(2).fill_(2)
        state.read_wb_fp8(unit, 0).fill_(1)
        state.acc[unit][0].fill_(3 if operation == 'matmul_acc' else 37)
        state.read_mrf_bf16(4).fill_(-1)
        state.read_mrf_bf16(5).fill_(-1)
        expected_value = {'matmul_acc': 35, 'weight_push': 2, 'acc_pop': 37}[operation]

        def result():
            if operation == 'matmul_acc':
                return state.acc[unit][0].float().clone()
            if operation == 'weight_push':
                return state.read_wb_fp8(unit, 0).float().clone()
            return state.read_mrf_bf16_tile(4).float().clone()

        def snapshot():
            values = result()
            return {
                'cycle': core.cycle_count,
                'active_engines': [engine.name for engine in core.exus if engine.has_in_flight],
                'correct_elements': int((values == expected_value).sum()),
                'correct_rows': int((values == expected_value).all(dim=1).sum()),
                'values': values.tolist(),
            }

        publication = None
        completion_cycle = None
        write_csrf = state.write_csrf

        def observe_csr_write(address, value):
            nonlocal publication
            write_csrf(address, value)
            if publication is None and address == 0xC10 and state.read_csrf(0xC10) == 1:
                # Observe before Matrix0/1 advance later in this tick.
                publication = snapshot()
                if operation == 'weight_push':
                    # Model source reuse without assuming host MREG access.
                    state.read_mrf_fp8(2).fill_(7)

        state.write_csrf = observe_csr_write
        with contextlib.redirect_stdout(io.StringIO()):
            while not core.is_finished() and core.cycle_count < 1000:
                core.tick()
                if completion_cycle is None and bool((result() == expected_value).all()):
                    completion_cycle = core.cycle_count
        assert core.is_finished(), 'MXU publication test timed out'
        assert publication is not None, 'never observed dbg0 == 1'
        observed = {
            'unit': unit,
            'operation': operation,
            'expected_value': expected_value,
            'expected_elements': 1024,
            'publication': publication,
            'completion_cycle': completion_cycle,
            'final': snapshot(),
            'source_reused_at_publication': operation == 'weight_push',
            'observation': 'first expected CSR write, before later engine ticks',
        }
        (directory / 'first-publication.json').write_text(json.dumps(observed, indent=2) + '\n')
        return observed
    finally:
        sim.close()


@pytest.mark.parametrize('unit', ['mxu0', 'mxu1'])
@pytest.mark.parametrize('operation', ['matmul_acc', 'acc_pop', 'weight_push'])
def test_release_completes_mxu_and_preserves_source_lifetime(
    publication_compiler, hardware_config_cls, publication_dir, unit, operation
):
    instruction = {
        'matmul_acc': f'vmatmul.acc.{unit} acc0, m0, w0',
        'acc_pop': f'vmatpop.bf16.acc.{unit} m4, acc0',
        'weight_push': f'vmatpush.weight.{unit} w0, m2',
    }[operation]
    source = instruction + '\ndelay 128\n' + PUBLISH
    optimized = publication_compiler(source, publication_dir)
    before = observe_mxu(source, hardware_config_cls, publication_dir / 'reference', unit, operation)
    after = observe_mxu(optimized, hardware_config_cls, publication_dir / 'optimized', unit, operation)

    # Expect matmul 32*(1*1)+3, BF16 pop 37, and FP8 push 2.
    # Reusing the source after release must preserve the pushed weights.
    assert before['publication']['correct_elements'] == 1024
    assert before['final']['correct_elements'] == 1024
    assert before['publication']['active_engines'] == []
    assert after['final']['correct_elements'] == 1024, 'source reused before MXU finished reading it'
    assert after['publication']['correct_elements'] == 1024, 'MXU output incomplete at first publication'
    assert after['publication']['active_engines'] == [], 'MXU active at first publication'
    assert after['publication']['values'] == before['publication']['values']
    assert after['completion_cycle'] < after['publication']['cycle']

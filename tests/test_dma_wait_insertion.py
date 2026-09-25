"""Repair missing waits, using explicit-wait programs as the safe references."""

import contextlib
from dataclasses import replace
import hashlib
import io
import json

import pytest

from tests import harness
from tests.test_publication import (
    PUBLISH,
    assert_safe_publication,
    observe,
    publication_compiler,
    publication_dir,
)


def run(source, hardware_config_cls, directory, *, dma_scale):
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'program.S').write_text(source)
    config = hardware_config_cls()
    config.arch_state_config = replace(config.arch_state_config, dram_size=16384)
    config.offchip_link_core_cycles_per_beat *= dma_scale
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
        state.mrf[0].fill_(0x31)
        state.vmem[:1024].fill_(0x55)
        state.dram[:1024].fill_(0xC3)
        state.dram[1024:2048].fill_(0x97)
        with contextlib.redirect_stdout(io.StringIO()):
            while not core.is_finished() and core.cycle_count < 100000:
                core.tick()
        assert core.is_finished(), 'DMA wait test timed out'
        result = {
            'xrf': tuple(state.xrf),
            'vmem': state.vmem.numpy().tobytes(),
            'dram': state.dram.numpy().tobytes(),
            'mreg6': state.mrf[6].numpy().tobytes(),
            'dma_base': state.base,
            'halt': state.halt_reason,
        }
        summary = {key: hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else value
                   for key, value in result.items()}
        summary.update(cycles=core.cycle_count, dma_scale=dma_scale)
        (directory / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        return result
    finally:
        sim.close()


def compare_repaired(source, reference, compiler, config_cls, directory, *, dma_scale):
    # Missing-wait input has no trustworthy asynchronous execution to compare with.
    expected = run(reference, config_cls, directory / 'reference', dma_scale=dma_scale)
    optimized = compiler(source, directory)
    actual = run(optimized, config_cls, directory / 'optimized', dma_scale=dma_scale)
    assert actual == expected
    repeat = directory / 'repeat'
    repeat.mkdir(exist_ok=True)
    assert compiler(optimized, repeat) == optimized
    return optimized, actual


@pytest.mark.parametrize('dma_scale', [1, 100])
@pytest.mark.parametrize('setup,launch,consumer,reference_wait,expected_region,offset,expected', [
    ('addi x7, x0, 32\n', 'dma.load.ch0 x0, x0, x7\n',
     'lw x2, 0(x0)\ndelay 4\nsw x2, 1024(x0)\n', 'dma.wait.ch0\n',
     'vmem', 1024, bytes([0xC3]) * 4),
    ('addi x7, x0, 1024\n', 'dma.load.ch0 x0, x0, x7\n',
     'vload m6, 0(x0)\ndelay 40\n', 'dma.wait.ch0\n',
     'mreg6', 0, bytes([0xC3]) * 1024),
    ('addi x7, x0, 32\naddi x2, x0, 85\n', 'dma.load.ch0 x0, x0, x7\n',
     'sw x2, 0(x0)\n', 'dma.wait.ch0\n', 'vmem', 0, bytes([85, 0, 0, 0])),
    ('addi x7, x0, 32\n', 'dma.store.ch0 x0, x0, x7\n',
     'vstore m0, 0(x0)\ndelay 40\n', 'dma.wait.ch0\n',
     'dram', 0, bytes([0x55]) * 32),
    ('addi x7, x0, 32\naddi x8, x0, 1536\n', 'dma.load.ch0 x0, x0, x7\n',
     'dma.store.ch1 x8, x0, x7\ndma.wait.ch1\n', 'dma.wait.ch0\n',
     'dram', 1536, bytes([0xC3]) * 32),
], ids=['scalar-read', 'vector-read', 'scalar-write', 'store-source-lifetime', 'cross-channel-overlap'])
def test_wait_precedes_conflicting_vmem_access(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
    setup, launch, consumer, reference_wait, expected_region, offset, expected,
):
    optimized, result = compare_repaired(
        setup + launch + consumer, setup + launch + reference_wait + consumer,
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert 'dma.wait.ch0' in optimized
    assert result[expected_region][offset:offset + len(expected)] == expected


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_disjoint_vmem_work_can_run_before_wait(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    source = ('addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\n'
              'vstore m0, 0(x0)\ndelay 40\n' + PUBLISH)
    reference = source.replace(PUBLISH, 'dma.wait.ch0\n' + PUBLISH)
    before = observe(reference, hardware_config_cls, publication_dir / 'reference', dma_scale=dma_scale)
    optimized = publication_compiler(source, publication_dir)
    after = observe(optimized, hardware_config_cls, publication_dir / 'optimized', dma_scale=dma_scale)
    assert_safe_publication(before, after)
    assert optimized.index('vstore') < optimized.index('dma.wait.ch0')
    assert after['vmem'][:1024] == bytes([0x31]) * 1024
    assert after['vmem'][4096:4128] == bytes([0xC3]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
@pytest.mark.parametrize('launch,updates,waits,expected_offset', [
    ('dma.load.ch0 x1, x0, x7\n', 'addi x1, x1, 32\naddi x7, x0, 64\n',
     'dma.wait.ch0\n', 4096),
    ('dma.load.ch1 x1, x0, x7\ndma.config.ch0 x8\n', 'addi x8, x0, 1\n',
     'dma.wait.ch0\n', 4096),
], ids=['load-operands', 'configuration-source'])
def test_wait_protects_dma_operands_read_at_completion(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
    launch, updates, waits, expected_offset,
):
    setup = 'addi x7, x0, 32\nlui x1, 1\naddi x8, x0, 0\n'
    tail = 'dma.wait.ch1\n' if 'dma.config' in launch else ''
    _, result = compare_repaired(
        setup + launch + updates, setup + launch + waits + updates + tail,
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert result['vmem'][expected_offset:expected_offset + 32] == bytes([0xC3]) * 32
    assert result['dma_base'] == 0


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_same_channel_reuse_is_repaired_without_a_release_marker(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    setup = 'addi x7, x0, 32\nlui x1, 1\nlui x2, 2\naddi x8, x0, 1024\n'
    first = 'dma.load.ch0 x1, x0, x7\n'
    second = 'dma.load.ch0 x2, x8, x7\n'
    optimized, result = compare_repaired(
        setup + first + second, setup + first + 'dma.wait.ch0\n' + second + 'dma.wait.ch0\n',
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert optimized.count('dma.wait.ch0') == 2
    assert result['vmem'][4096:4128] == bytes([0xC3]) * 32
    assert result['vmem'][8192:8224] == bytes([0x97]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_disjoint_channels_and_base_configuration_preserve_dma_queue_order(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    setup = ('addi x7, x0, 32\nlui x1, 1\nlui x2, 2\n'
             'addi x8, x0, 1\naddi x9, x0, 1024\n')
    commands = ('dma.load.ch0 x1, x0, x7\ndma.load.ch2 x2, x9, x7\n'
                'dma.config.ch1 x8\n')
    reference = setup + commands + 'dma.wait.ch0\ndma.wait.ch1\ndma.wait.ch2\n'
    optimized, result = compare_repaired(
        setup + commands, reference, publication_compiler, hardware_config_cls,
        publication_dir, dma_scale=dma_scale,
    )
    assert result['vmem'][4096:4128] == bytes([0xC3]) * 32
    assert result['vmem'][8192:8224] == bytes([0x97]) * 32
    assert result['dma_base'] == 1
    # A base-register change is ordered by the DMA queue, so unrelated channels
    # can all launch before the exit waits rather than serializing transfers.
    assert optimized.index('dma.config.ch1') < optimized.index('dma.wait.')


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_unknown_scalar_address_conservatively_waits_for_dma(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    setup = ('addi x7, x0, 32\nlui x1, 1\nsw x1, 1024(x0)\n'
             'lw x2, 1024(x0)\ndelay 4\n')
    launch = 'dma.load.ch0 x1, x0, x7\n'
    consumer = 'lw x3, 0(x2)\ndelay 4\nsw x3, 1536(x0)\n'
    _, result = compare_repaired(
        setup + launch + consumer, setup + launch + 'dma.wait.ch0\n' + consumer,
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert result['vmem'][1536:1540] == bytes([0xC3]) * 4


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_overlapping_dma_stores_serialize_even_when_both_only_read_vmem(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    setup = 'addi x7, x0, 32\naddi x8, x0, 1024\n'
    first = 'dma.store.ch0 x0, x0, x7\n'
    second = 'dma.store.ch1 x8, x0, x7\n'
    _, result = compare_repaired(
        setup + first + second + 'ecall\n',
        setup + first + 'dma.wait.ch0\n' + second + 'dma.wait.ch1\necall\n',
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert result['dram'][:32] == bytes([0x55]) * 32
    assert result['dram'][1024:1056] == bytes([0x55]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
@pytest.mark.parametrize('branch_value,expected', [(0, 17), (1, 29)],
                         ids=['fallthrough', 'taken'])
def test_dma_operand_write_in_retained_delay_slot_preserves_branch_result(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale, branch_value, expected,
):
    setup = f'addi x7, x0, 32\nlui x1, 1\nlui x9, {branch_value}\n'
    launch = 'dma.load.ch0 x1, x0, x7\n'
    branch = ('beq x1, x9, taken\naddi x1, x1, 32\n'
              'addi x2, x0, 17\njal x0, done\nnop\n'
              'taken:\naddi x2, x0, 29\ndone:\nsw x2, 0(x0)\n')
    _, result = compare_repaired(
        setup + launch + branch, setup + launch + 'dma.wait.ch0\n' + branch,
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert result['xrf'][1] == 4128
    assert result['xrf'][2] == expected
    assert result['vmem'][4096:4128] == bytes([0xC3]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
@pytest.mark.parametrize('ending', ['', 'ecall\n', 'ebreak\n', 'end:\n'],
                         ids=['falloff', 'ecall', 'ebreak', 'empty-exit-label'])
def test_wait_drains_dma_before_program_exit(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale, ending,
):
    prefix = 'addi x7, x0, 32\ndma.store.ch0 x0, x0, x7\n'
    optimized, result = compare_repaired(
        prefix + ending, prefix + 'dma.wait.ch0\n' + ending,
        publication_compiler, hardware_config_cls, publication_dir, dma_scale=dma_scale,
    )
    assert 'dma.wait.ch0' in optimized
    assert result['dram'][:32] == bytes([0x55]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
@pytest.mark.parametrize('branch_value', [0, 1], ids=['waited-path', 'unwaited-path'])
def test_release_repairs_join_with_only_one_waited_predecessor(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale, branch_value,
):
    source = (
        f'addi x9, x0, {branch_value}\naddi x7, x0, 32\nlui x1, 1\n'
        'dma.load.ch0 x1, x0, x7\nbeq x9, x0, waited\nnop\n'
        'jal x0, publish\nnop\nwaited:\ndma.wait.ch0\n'
        'publish:\n' + PUBLISH
    )
    reference = source.replace(PUBLISH, 'dma.wait.ch0\n' + PUBLISH)
    before = observe(reference, hardware_config_cls, publication_dir / 'reference', dma_scale=dma_scale)
    optimized = publication_compiler(source, publication_dir)
    after = observe(optimized, hardware_config_cls, publication_dir / 'optimized', dma_scale=dma_scale)
    assert_safe_publication(before, after)
    assert after['vmem'][4096:4128] == bytes([0xC3]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_wait_repairs_reuse_carried_by_loop_backedge(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    setup = 'addi x7, x0, 32\nlui x1, 1\naddi x10, x0, 0\naddi x11, x0, 3\n'
    body = ('loop:\ndma.load.ch0 x1, x0, x7\naddi x10, x10, 1\n'
            'blt x10, x11, loop\nnop\n')
    reference = setup + body.replace('addi x10, x10, 1', 'dma.wait.ch0\naddi x10, x10, 1')
    _, result = compare_repaired(
        setup + body, reference, publication_compiler, hardware_config_cls,
        publication_dir, dma_scale=dma_scale,
    )
    assert result['xrf'][10] == 3
    assert result['vmem'][4096:4128] == bytes([0xC3]) * 32


@pytest.mark.parametrize('dma_scale', [1, 100])
def test_wait_repairs_dma_carried_back_to_entry_release(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale,
):
    source = (
        'entry:\naddi x10, x10, 1\ncsrrw x0, x10, 0xC10 # atlas.release\n'
        'addi x11, x0, 3\nbge x10, x11, done\nnop\n'
        'addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\n'
        'jal x0, entry\nnop\ndone:\naddi x12, x0, 9\n'
    )
    reference = source.replace('entry:\n', 'entry:\ndma.wait.ch0\n')
    before = observe(reference, hardware_config_cls, publication_dir / 'reference',
                     dma_scale=dma_scale, expected_dbg0=3)
    optimized = publication_compiler(source, publication_dir)
    after = observe(optimized, hardware_config_cls, publication_dir / 'optimized',
                    dma_scale=dma_scale, expected_dbg0=3)
    assert_safe_publication(before, after)
    assert after['vmem'][4096:4128] == bytes([0xC3]) * 32

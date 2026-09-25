"""Check publication across branches and loops."""

import pytest

from tests import harness
from tests.test_publication import (
    PUBLISH,
    assert_safe_publication,
    compare,
    observe,
    publication_compiler,
    publication_dir,
    publication_without_dma_insertion,
)


@pytest.mark.parametrize('branch_value', [0, 1], ids=['taken', 'fallthrough'])
def test_release_after_branch_with_unfinished_fixed_work(
    publication_compiler, hardware_config_cls, publication_dir, branch_value
):
    # Branch during VSTORE; distinct per-path results verify which path ran.
    source = (
        f'addi x9, x0, {branch_value}\nvstore m0, 0(x0)\n'
        'beq x9, x0, taken\nnop\n'
        'vmov m8, m4\ndelay 65\njal x0, publish\nnop\n'
        'taken:\nvtrpose.xlu m6, m2\ndelay 66\n'
        'publish:\ndelay 40\n' + PUBLISH
    )
    _, before, after = compare(source, publication_compiler, hardware_config_cls, publication_dir)
    assert before['vmem'][:1024] == after['vmem'][:1024] == bytes([0x31]) * 1024
    if branch_value == 0:
        assert after['mreg6'] == bytes([0x7B]) * 1024
        assert after['mreg8'] == bytes(1024)
    else:
        assert after['mreg8'] == bytes([0x2A]) * 1024
        assert after['mreg6'] == bytes(1024)


@pytest.mark.parametrize('dma_scale', [1, 10])
def test_release_on_third_iteration_after_matching_dma_wait(
    publication_compiler, hardware_config_cls, publication_dir, dma_scale
):
    # Observe iteration 3, after two backedges.
    source = (
        'addi x7, x0, 32\nlui x1, 1\naddi x10, x0, 0\naddi x11, x0, 3\n'
        'loop:\ndma.load.ch0 x1, x0, x7\nvstore m0, 0(x0)\ndma.wait.ch0\n'
        'delay 40\naddi x10, x10, 1\n'
        'csrrw x0, x10, 0xC10 # atlas.release\n'
        'blt x10, x11, loop\nnop\n'
    )
    optimized = publication_compiler(source, publication_dir)
    before = observe(source, hardware_config_cls, publication_dir / 'reference',
                     dma_scale=dma_scale, expected_dbg0=3)
    after = observe(optimized, hardware_config_cls, publication_dir / 'optimized',
                    dma_scale=dma_scale, expected_dbg0=3)
    assert_safe_publication(before, after)
    assert after['vmem'][:1024] == bytes([0x31]) * 1024
    assert after['vmem'][4096:4128] == bytes([0xC3]) * 32


def _must_reject_pending_path(source, compiler, hardware_config_cls, directory, *, expected_dbg0=1):
    # This delay covers 10x DMA, not arbitrary latency. Record legacy acceptance.
    before = observe(source, hardware_config_cls, directory / 'reference',
                     dma_scale=10, expected_dbg0=expected_dbg0)
    assert before['active_engines'] == ()
    try:
        optimized = compiler(source, directory)
    except harness.OptimizerError as error:
        assert 'atlas.release' in str(error) and 'pending DMA' in str(error)
        assert not (directory / 'after.S').exists()
        return
    after = observe(optimized, hardware_config_cls, directory / 'accepted-optimized',
                    dma_scale=10, expected_dbg0=expected_dbg0)
    pytest.fail(
        'compiler accepted a release with a potentially pending DMA path; '
        f'publication cycle={after["cycle"]}, active_engines={after["active_engines"]}'
    )


@pytest.mark.parametrize('branch_value', [0, 1], ids=['waited-path', 'unwaited-path'])
def test_release_rejects_join_with_only_one_waited_predecessor(
    publication_without_dma_insertion, hardware_config_cls, publication_dir, branch_value
):
    # Reject both inputs: every reaching path needs a matching wait.
    source = (
        f'addi x9, x0, {branch_value}\naddi x7, x0, 32\nlui x1, 1\n'
        'dma.load.ch0 x1, x0, x7\nbeq x9, x0, waited\nnop\n'
        'jal x0, publish\nnop\nwaited:\ndma.wait.ch0\n'
        'publish:\ndelay 300\n' + PUBLISH
    )
    _must_reject_pending_path(source, publication_without_dma_insertion,
                             hardware_config_cls, publication_dir)


def test_release_rejects_dma_carried_back_to_entry(
    publication_without_dma_insertion, hardware_config_cls, publication_dir
):
    # First entry is idle; the backedge carries DMA into the second release.
    source = (
        'entry:\naddi x10, x10, 1\ndelay 300\n'
        'csrrw x0, x10, 0xC10 # atlas.release\n'
        'addi x11, x0, 2\nbge x10, x11, done\nnop\n'
        'addi x7, x0, 32\nlui x1, 1\ndma.load.ch0 x1, x0, x7\n'
        'jal x0, entry\nnop\ndone:\naddi x12, x0, 9\n'
    )
    _must_reject_pending_path(source, publication_without_dma_insertion, hardware_config_cls,
                             publication_dir, expected_dbg0=2)

"""Check that releases require waits before channel reuse."""

import json

import pytest

from tests import harness
from tests.test_publication import (
    PUBLISH,
    compare,
    observe,
    publication_compiler,
    publication_dir,
)


REUSED_CHANNEL = (
    'dma.config.ch0 x0\ndelay 100 # keep\n'
    'dma.config.ch0 x0\ndma.wait.ch0\n' + PUBLISH
)
REUSED_LOAD_CHANNEL = (
    'addi x7, x0, 32\nlui x1, 1\nlui x2, 2\n'
    'dma.load.ch0 x1, x0, x7\ndelay 1000 # keep\n'
    'dma.load.ch0 x2, x0, x7\ndma.wait.ch0\n' + PUBLISH
)


def _observe_status(source, hardware_config_cls, directory, dma_scale):
    try:
        value = observe(source, hardware_config_cls, directory, dma_scale=dma_scale)
        result = {'result': 'published', 'cycle': value['cycle'],
                  'active_engines': value['active_engines']}
    except AssertionError as error:
        result = {'result': 'assertion', 'message': str(error)}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'reuse-status.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


@pytest.mark.parametrize('source', [REUSED_CHANNEL, REUSED_LOAD_CHANNEL],
                         ids=['model-config', 'real-load'])
def test_release_rejects_channel_reuse_hidden_by_fixed_delay(
    publication_compiler, hardware_config_cls, publication_dir, source
):
    # Fixed delays can hide reuse before the channel's busy flag clears.
    for scale in [1, 10]:
        reference = _observe_status(source, hardware_config_cls,
                                    publication_dir / f'reference-{scale}x', scale)
        assert reference['result'] == 'published'
        assert reference['active_engines'] == ()
    slow = _observe_status(source, hardware_config_cls,
                           publication_dir / 'reference-100x', 100)
    assert slow['result'] == 'assertion'
    assert 'Flag 0 is already set' in slow['message']

    try:
        optimized = publication_compiler(source, publication_dir)
    except harness.OptimizerError as error:
        assert 'atlas.release' in str(error)
        assert 'ch0' in str(error)
        assert not (publication_dir / 'after.S').exists()
        return
    # Record legacy acceptance and its 100x failure.
    after = _observe_status(optimized, hardware_config_cls,
                            publication_dir / 'accepted-optimized-100x', 100)
    pytest.fail(f'compiler accepted a possibly busy channel reuse: {after}')


def test_release_accepts_channel_reuse_after_each_matching_wait(
    publication_compiler, hardware_config_cls, publication_dir
):
    # Matching waits permit config/load reuse at 100x latency.
    source = ('addi x7, x0, 32\nlui x1, 1\n'
              'dma.config.ch0 x0\ndma.wait.ch0\n'
              'dma.load.ch0 x1, x0, x7\ndma.wait.ch0\n' + PUBLISH)
    optimized, before, after = compare(source, publication_compiler, hardware_config_cls,
                                      publication_dir, dma_scale=100)
    assert optimized.count('dma.wait.ch0') == 2
    assert before['vmem'][4096:4128] == after['vmem'][4096:4128] == bytes([0xC3]) * 32


def test_unmarked_channel_reuse_retains_legacy_acceptance(
    publication_compiler, publication_dir
):
    # Legacy acceptance is unchanged; this stream still fails at 100x latency.
    source = REUSED_CHANNEL.replace('# atlas.release', '# progress only')
    optimized = publication_compiler(source, publication_dir)
    assert 'atlas.release' not in optimized
    assert optimized.count('dma.config.ch0') == 2
    assert 'keep' in optimized

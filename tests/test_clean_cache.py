"""Tests for `fastpose-clean-cache`.

The tool deletes files, so what matters is that it deletes exactly the numba
kernel cache and nothing adjacent to it.
"""

import os

import pytest

from fastpose import clean_cache


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A directory holding a plausible mix of cache and non-cache files."""
    cache_dir = tmp_path / '__pycache__'
    cache_dir.mkdir()
    kernels = {}
    for name, size in (('ransac.driver-3.py310.nbi', 100),
                       ('ransac.driver-3.py310.1.nbc', 4096),
                       ('lm.refine-7.py311.nbi', 100),
                       ('lm.refine-7.py311.2.nbc', 2048)):
        (cache_dir / name).write_bytes(b'x' * size)
        kernels[name] = size

    # things that must survive: CPython bytecode, sources, anything else
    survivors = ['ransac.cpython-310.pyc', 'lm.cpython-310.pyc', 'notes.txt']
    for name in survivors:
        (cache_dir / name).write_bytes(b'keep')
    (tmp_path / 'ransac.py').write_bytes(b'keep')

    monkeypatch.setattr(clean_cache, 'cache_dirs', lambda: [str(cache_dir)])
    return tmp_path, cache_dir, kernels, survivors


def test_finds_only_kernel_cache_files(fake_cache):
    _, cache_dir, kernels, _ = fake_cache
    found = clean_cache.find_cache_files()

    assert {os.path.basename(p) for p, _ in found} == set(kernels)
    # reported sizes are real, and the listing is largest first
    assert dict((os.path.basename(p), s) for p, s in found) == kernels
    assert [s for _, s in found] == sorted(kernels.values(), reverse=True)


def test_dry_run_reports_but_deletes_nothing(fake_cache):
    _, cache_dir, kernels, _ = fake_cache
    before = sorted(os.listdir(cache_dir))

    removed, freed = clean_cache.clean(dry_run=True)

    assert removed == len(kernels)
    assert freed == sum(kernels.values())
    assert sorted(os.listdir(cache_dir)) == before


def test_clean_removes_kernels_and_spares_everything_else(fake_cache):
    tmp_path, cache_dir, kernels, survivors = fake_cache

    removed, freed = clean_cache.clean()

    assert removed == len(kernels)
    assert freed == sum(kernels.values())
    assert sorted(os.listdir(cache_dir)) == sorted(survivors)
    # the source next to the cache directory is untouched, and the directory
    # itself stays: it still holds CPython's bytecode
    assert (tmp_path / 'ransac.py').exists()
    assert cache_dir.is_dir()


def test_clean_is_idempotent(fake_cache):
    clean_cache.clean()
    removed, freed = clean_cache.clean()
    assert (removed, freed) == (0, 0)


def test_cache_dirs_cover_every_package_directory():
    # every directory of the installed package that holds sources must be
    # searched, or a cleanup silently misses a subpackage
    dirs = {os.path.normcase(os.path.abspath(d)) for d in clean_cache.cache_dirs()}
    for source_dir in clean_cache._package_dirs():
        in_tree = os.path.join(source_dir, '__pycache__')
        assert os.path.normcase(os.path.abspath(in_tree)) in dirs


def test_cache_dirs_follows_numba_cache_dir(monkeypatch, tmp_path):
    # NUMBA_CACHE_DIR is read from numba's config, and the subdirectory name is
    # numba's own hash of the source path - not something reimplemented here
    from numba import config
    from numba.core.caching import _CacheLocator

    monkeypatch.setattr(config, 'CACHE_DIR', str(tmp_path), raising=False)
    dirs = clean_cache.cache_dirs()

    source_dir = clean_cache._package_dirs()[0]
    expected = os.path.join(
        str(tmp_path),
        _CacheLocator.get_suitable_cache_subpath(os.path.join(source_dir, 'x.py')))
    assert expected in dirs


def test_main_dry_run_exits_clean(fake_cache, capsys):
    assert clean_cache.main(['--dry-run']) == 0
    out = capsys.readouterr().out
    assert 'Would remove 4 file(s)' in out


def test_main_reports_when_there_is_nothing_to_do(fake_cache, capsys):
    clean_cache.clean()
    assert clean_cache.main([]) == 0
    assert 'No cached kernels found.' in capsys.readouterr().out

"""Command-line removal of fastpose's cached Numba kernels.

`fastpose-warmup` writes a lot: a full warmup produces hundreds of MB of
`.nbc`/`.nbi` files, and nothing ever reclaims them. Superseded entries are not
overwritten in place either - the index is keyed on each source file's
timestamp and size and on the host CPU, so editing a kernel, upgrading numba or
upgrading fastpose orphans the old entries rather than replacing them, and pip
does not remove them on upgrade because they are generated at runtime and never
appear in the wheel's RECORD.

Numba may put them in any of three places depending on how fastpose is
installed, which is why finding them by hand is awkward:

    NUMBA_CACHE_DIR   if that variable is set (UserProvidedCacheLocator)
    <package>/**/__pycache__   the usual case  (InTreeCacheLocator)
    a user-wide cache directory, when the install directory is not writable,
                      e.g. a system or container install (UserWideCacheLocator)

This checks all three. It only ever deletes `.nbc` and `.nbi` files that sit in
a cache directory belonging to a fastpose source directory: never `.py`, never
`.pyc` (those are CPython's own and are cheap to rebuild), and never anything
under another package. Deleting them is always safe - the kernels recompile on
next use, or run `fastpose-warmup` to rebuild them up front.
"""

import argparse
import os

import fastpose

# what numba writes: the per-signature index and the compiled data files
CACHE_SUFFIXES = ('.nbi', '.nbc')


def _package_dirs():
    """Every directory of the installed fastpose package that holds sources."""
    root = os.path.dirname(os.path.abspath(fastpose.__file__))
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        if any(f.endswith('.py') for f in filenames):
            dirs.append(dirpath)
    return dirs


def cache_dirs():
    """Cache directories numba could have used for fastpose's kernels.

    Mirrors the three locators in `numba.core.caching` that apply to a package
    backed by real source files, using numba's own subpath helper so this keeps
    agreeing with it. Directories that do not exist are still returned; callers
    filter them.
    """
    from numba import config
    from numba.core.caching import _CacheLocator

    candidates = []
    bases = []
    if getattr(config, 'CACHE_DIR', None):
        bases.append(config.CACHE_DIR)
    try:
        from numba.misc.appdirs import AppDirs
        bases.append(AppDirs(appname='numba', appauthor=False).user_cache_dir)
    except Exception:  # pragma: no cover - numba internals moved
        pass

    for source_dir in _package_dirs():
        # in-tree: the __pycache__ beside the sources
        candidates.append(os.path.join(source_dir, '__pycache__'))
        # the redirected and user-wide variants, which hash the source path
        marker = os.path.join(source_dir, 'x.py')
        for base in bases:
            subpath = _CacheLocator.get_suitable_cache_subpath(marker)
            candidates.append(os.path.join(base, subpath))

    seen = set()
    unique = []
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_cache_files():
    """(path, size) of every fastpose kernel cache file, largest first."""
    found = []
    for directory in cache_dirs():
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.endswith(CACHE_SUFFIXES):
                continue
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                found.append((path, os.path.getsize(path)))
    found.sort(key=lambda item: -item[1])
    return found


def _human(num_bytes):
    size = float(num_bytes)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024.0 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024.0


def clean(dry_run=False, verbose=False):
    """Delete the cache files; returns (num_removed, bytes_removed)."""
    files = find_cache_files()
    removed = 0
    freed = 0
    for path, size in files:
        if dry_run:
            removed += 1
            freed += size
            if verbose:
                print(f'  would remove {path} ({_human(size)})')
            continue
        try:
            os.remove(path)
        except OSError as exc:
            print(f'  could not remove {path}: {exc}')
            continue
        removed += 1
        freed += size
        if verbose:
            print(f'  removed {path} ({_human(size)})')
    return removed, freed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove fastpose's cached Numba kernels (.nbi/.nbc). They "
                    "recompile on next use; `fastpose-warmup` rebuilds them "
                    "up front.",
    )
    parser.add_argument(
        '-n', '--dry-run', action='store_true',
        help='Report what would be removed without deleting anything.',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='List every file rather than just the total.',
    )
    parser.add_argument(
        '--list-dirs', action='store_true',
        help='Print the cache directories that were searched and exit.',
    )
    args = parser.parse_args(argv)

    if args.list_dirs:
        for directory in cache_dirs():
            mark = '' if os.path.isdir(directory) else '  (does not exist)'
            print(f'{directory}{mark}')
        return 0

    removed, freed = clean(dry_run=args.dry_run, verbose=args.verbose)
    if not removed:
        print('No cached kernels found.')
        return 0
    verb = 'Would remove' if args.dry_run else 'Removed'
    print(f'{verb} {removed} file(s), {_human(freed)}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

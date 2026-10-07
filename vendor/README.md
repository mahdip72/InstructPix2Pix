# Patched Accelerate 1.15.0

`accelerate-1.15.0+instructpix2pix.1-py3-none-any.whl` is a locally patched
distribution of upstream Accelerate 1.15.0. The Apache 2.0 license, package
contents and command entry points are retained in the wheel.

Upstream wheel SHA256:
`97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec`

Patched wheel SHA256:
`3007207fba7d1a14dcf441486b412502f6975a42cba50b532b6c61fbf2ef2e0d`

The [upstream advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79)
has no patched release. The 1.15.0 source still accepts unsafe shard paths.
`accelerate-security.patch` validates the entire index before any shard load:
contained relative paths, nonempty string names, canonical containment, and
regular files for both index and shards. It rejects traversal, drive/UNC paths,
symlink escapes, named pipes, sockets, devices, directories and missing shards.
Directory-selected whole checkpoints receive the same containment and regular-file
checks. The central loader is also used by `load_checkpoint_and_dispatch`.

Direct Accelerate directory/index loads now require real files contained in the checkpoint
directory. Hub snapshot symlinks into sibling `blobs` must be copied into a
self-contained checkpoint directory before using those two Accelerate loaders.
The project's Diffusers/Transformers component loaders and Accelerator resume
API do not use this sharded loader. Do not replace the local wheel with upstream
1.15.0 or disable validation to accept a malformed checkpoint. Checkpoint
directories should remain immutable while loading.

To reproduce the wheel, download the upstream `accelerate-1.15.0-py3-none-any.whl`
linked from [PyPI release metadata](https://pypi.org/pypi/accelerate/1.15.0/json),
then run from the repository root:

```sh
python vendor/build_accelerate_wheel.py /path/to/accelerate-1.15.0-py3-none-any.whl
python -m unittest discover -s tests -v
```

The builder verifies the upstream hash, applies exactly one source block change,
updates the local version and regenerates wheel hashes. Replace this vendor
patch only after confirming that an upstream release fixes the issue and the
regression tests pass. The named-pipe regression runs on POSIX and skips on Windows.

"""Rebuild the hash-verified upstream wheel with only the recorded security patch."""
import argparse
import base64
import csv
import difflib
import hashlib
import io
from pathlib import Path
import zipfile

UPSTREAM_SHA256 = "97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec"
VERSION = "1.15.0+instructpix2pix.1"
OLD_BLOCK = '''    if index_filename is not None:
        checkpoint_folder = os.path.split(index_filename)[0]
        with open(index_filename) as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]
'''
NEW_BLOCK = '''    if index_filename is not None:
        # Local security patch for GHSA-4j2p-28q2-5m79. Validate every
        # shard before opening any of them, including POSIX named pipes.
        import ntpath
        import stat

        checkpoint_folder = os.path.realpath(os.path.dirname(index_filename) or ".")
        index_path = os.path.realpath(index_filename)
        if (os.path.commonpath([checkpoint_folder, index_path]) != checkpoint_folder
                or not stat.S_ISREG(os.stat(index_path).st_mode)):
            raise ValueError("Checkpoint index must be a regular file inside its directory")
        with open(index_path) as f:
            index = json.load(f)
        if not isinstance(index, dict):
            raise ValueError("Checkpoint index must contain a nonempty weight map")
        index = index.get("weight_map", index)
        if not isinstance(index, dict) or not index:
            raise ValueError("Checkpoint index must contain a nonempty weight map")

        checkpoint_files = []
        for name in index.values():
            if (not isinstance(name, str) or not name or "\\0" in name
                    or ":" in name or os.path.isabs(name) or ntpath.splitdrive(name)[0]
                    or name.startswith("\\\\") or ".." in name.replace("\\\\", "/").split("/")):
                raise ValueError("Checkpoint shards must use contained relative paths")
            shard = os.path.realpath(os.path.join(checkpoint_folder, *name.replace("\\\\", "/").split("/")))
            try:
                contained = os.path.commonpath([checkpoint_folder, shard]) == checkpoint_folder
                regular = stat.S_ISREG(os.stat(shard).st_mode)
            except (OSError, ValueError) as error:
                raise ValueError("Checkpoint shard is missing or invalid") from error
            if not contained or not regular:
                raise ValueError("Checkpoint shard must be a regular file inside its directory")
            checkpoint_files.append(shard)
        checkpoint_files = sorted(set(checkpoint_files))

    if index_filename is None and os.path.isdir(checkpoint):
        # Directory-selected whole checkpoints must obey the same boundary.
        import stat

        checkpoint_folder = os.path.realpath(checkpoint)
        validated_files = []
        for filename in checkpoint_files:
            filename = os.path.realpath(filename)
            try:
                contained = os.path.commonpath([checkpoint_folder, filename]) == checkpoint_folder
                regular = stat.S_ISREG(os.stat(filename).st_mode)
            except (OSError, ValueError) as error:
                raise ValueError("Checkpoint file is missing or invalid") from error
            if not contained or not regular:
                raise ValueError("Checkpoint file must be a regular file inside its directory")
            validated_files.append(filename)
        checkpoint_files = validated_files
'''


def build(upstream, output):
    wheel_bytes = upstream.read_bytes()
    if hashlib.sha256(wheel_bytes).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Upstream Accelerate wheel SHA256 does not match")
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as source:
        files = {name: source.read(name) for name in source.namelist()}
    path = "accelerate/utils/modeling.py"
    original = files[path].decode("utf-8")
    if original.count(OLD_BLOCK) != 1:
        raise ValueError("Expected upstream loader block was not found exactly once")
    patched = original.replace(OLD_BLOCK, NEW_BLOCK)
    compile(patched, path, "exec")
    files[path] = patched.encode("utf-8")
    init_path = "accelerate/__init__.py"
    files[init_path] = files[init_path].replace(b'__version__ = "1.15.0"',
                                             f'__version__ = "{VERSION}"'.encode())
    old_info = "accelerate-1.15.0.dist-info/"
    new_info = f"accelerate-{VERSION}.dist-info/"
    files = {name.replace(old_info, new_info): data for name, data in files.items()}
    metadata_path = new_info + "METADATA"
    files[metadata_path] = files[metadata_path].replace(b"Version: 1.15.0\n", f"Version: {VERSION}\n".encode())
    record_path = new_info + "RECORD"
    files.pop(record_path)
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record_path, "", ""])
    files[record_path] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"accelerate-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            target.writestr(entry, data)
    patch = "".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                       fromfile="a/" + path, tofile="b/" + path, n=0))
    (output / "accelerate-security.patch").write_text(patch, encoding="utf-8", newline="\n")
    print(wheel.name, hashlib.sha256(wheel.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build(args.upstream, args.output)

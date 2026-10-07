"""Offline regression checks for GHSA-4j2p-28q2-5m79; no pretrained models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from accelerate import load_checkpoint_and_dispatch
from accelerate.utils import load_checkpoint_in_model

torch.set_num_threads(1)


class CheckpointSecurityTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        self.index = self.checkpoint / "pytorch_model.bin.index.json"
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.weights = {"weight": torch.full((2, 2), 7.0)}
        torch.save(self.weights, self.root / "outside.bin")

    def write_index(self, values):
        self.index.write_text(json.dumps({"weight_map": values}), encoding="utf-8")

    def test_valid_local_shard_still_loads(self):
        nested = self.checkpoint / "nested"
        nested.mkdir()
        torch.save(self.weights, nested / "model.bin")
        self.write_index({"weight": "nested/model.bin"})
        for loader in (load_checkpoint_in_model, load_checkpoint_and_dispatch):
            with self.subTest(loader=loader.__name__):
                loader(self.model, str(self.index))
                torch.testing.assert_close(self.model.weight, self.weights["weight"])

    def test_single_file_checkpoint_still_loads(self):
        path = self.checkpoint / "model.bin"
        torch.save(self.weights, path)
        load_checkpoint_in_model(self.model, str(path))
        torch.testing.assert_close(self.model.weight, self.weights["weight"])

    def test_directory_single_file_checkpoint_still_loads(self):
        torch.save(self.weights, self.checkpoint / "pytorch_model.bin")
        load_checkpoint_in_model(self.model, str(self.checkpoint))
        torch.testing.assert_close(self.model.weight, self.weights["weight"])

    def test_rejects_single_file_directory_symlink_escape(self):
        link = self.checkpoint / "pytorch_model.bin"
        try:
            link.symlink_to(self.root / "outside.bin")
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            load_checkpoint_in_model(self.model, str(self.checkpoint))

    def test_parent_traversal_is_rejected_by_both_public_loaders(self):
        self.write_index({"weight": "../outside.bin"})
        for loader in (load_checkpoint_in_model, load_checkpoint_and_dispatch):
            with self.subTest(loader=loader.__name__), self.assertRaises(ValueError):
                loader(self.model, str(self.index))

    def test_rejects_absolute_and_cross_platform_paths(self):
        names = [str((self.root / "outside.bin").resolve()), "/etc/passwd",
                 "C:\\outside.bin", "C:outside.bin", "\\\\server\\share\\model.bin",
                 "..\\outside.bin", "nested/../../outside.bin", "model.bin\0"]
        for name in names:
            self.write_index({"weight": name})
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_checkpoint_in_model(self.model, str(self.index))

    def test_all_shards_are_validated_before_any_weight_changes(self):
        torch.save(self.weights, self.checkpoint / "a.bin")
        before = self.model.weight.detach().clone()
        self.write_index({"weight": "a.bin", "other": str((self.root / "outside.bin").resolve())})
        with self.assertRaises(ValueError):
            load_checkpoint_in_model(self.model, str(self.index))
        torch.testing.assert_close(self.model.weight, before)

    def test_rejects_directory_shards(self):
        (self.checkpoint / "directory.bin").mkdir()
        self.write_index({"weight": "directory.bin"})
        with self.assertRaises(ValueError):
            load_checkpoint_in_model(self.model, str(self.index))

    def test_rejects_missing_shards(self):
        self.write_index({"weight": "missing.bin"})
        with self.assertRaises(ValueError):
            load_checkpoint_in_model(self.model, str(self.index))

    def test_rejects_symlink_escape(self):
        link = self.checkpoint / "link.bin"
        try:
            link.symlink_to(self.root / "outside.bin")
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        self.write_index({"weight": link.name})
        with self.assertRaises(ValueError):
            load_checkpoint_in_model(self.model, str(self.index))

    def test_rejects_malformed_weight_maps(self):
        for mapping in ({}, [], {"weight": 42}, {"weight": ""}):
            self.write_index(mapping)
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                load_checkpoint_in_model(self.model, str(self.index))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX named pipes unavailable")
    def test_fifo_index_and_shard_fail_without_blocking(self):
        fifo = self.checkpoint / "pipe.bin"
        os.mkfifo(fifo)
        self.write_index({"weight": fifo.name})
        fifo_checkpoint = self.root / "fifo_checkpoint"
        fifo_checkpoint.mkdir()
        os.mkfifo(fifo_checkpoint / "pytorch_model.bin.index.json")
        single_checkpoint = self.root / "single_fifo_checkpoint"
        single_checkpoint.mkdir()
        os.mkfifo(single_checkpoint / "pytorch_model.bin")
        for path in (str(self.index), str(fifo_checkpoint), str(single_checkpoint)):
            code = ("import torch; from accelerate.utils import load_checkpoint_in_model; "
                    "load_checkpoint_in_model(torch.nn.Linear(2,2,bias=False), " + repr(path) + ")")
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError", result.stderr)


if __name__ == "__main__":
    unittest.main()

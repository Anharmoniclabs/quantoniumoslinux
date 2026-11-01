"""Storage faults retain the last committed local or remote checkpoint."""

from collections import OrderedDict
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

import phistate_checkpoint_io as io


class CheckpointIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.local = self.root / "runtime/latest.pt"
        self.local.parent.mkdir()
        self.remote = self.root / "remote/step_32.pt"
        self.payload = {
            "phase_step": 32,
            "history": [{"step": i} for i in range(1, 33)],
            "state_dict": OrderedDict(weight=torch.arange(8.0).reshape(2, 4)),
            "optimizer_state": {"state": {0: {"moment": torch.ones(2, 4)}}, "params": [0]},
            "rng": torch.get_rng_state(),
            "config": {"vocab": 8, "pair": (1, 2), "missing": None},
        }
        self.payload["state_dict"]._metadata = {"": {"version": 1}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_checkpoint_full_roundtrip_and_independent_cpu_snapshot(self):
        receipt = io.atomic_checkpoint(self.local, self.payload)
        self.assertEqual(receipt, io.file_receipt(self.local))
        loaded = torch.load(self.local, map_location="cpu", weights_only=True)
        self.assertEqual(loaded["phase_step"], 32)
        self.assertEqual(loaded["state_dict"]._metadata, self.payload["state_dict"]._metadata)
        self.assertEqual(loaded["state_dict"]["weight"].device.type, "cpu")
        self.payload["state_dict"]["weight"].fill_(999)
        self.assertEqual(loaded["state_dict"]["weight"][0, 0], 0)
        self.assertFalse(list(self.local.parent.glob("*.partial")))

    def test_local_full_disk_preserves_prior_checkpoint_and_unrelated_partial(self):
        original = io.atomic_checkpoint(self.local, self.payload)
        unrelated = self.local.parent / "other.partial"
        unrelated.write_bytes(b"do not touch")

        def failing_save(value, stream):
            stream.write(b"incomplete new file")
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch.object(io.torch, "save", side_effect=failing_save):
            with self.assertRaises(OSError) as error:
                io.atomic_checkpoint(self.local, {**self.payload, "phase_step": 64})
        self.assertEqual(error.exception.errno, errno.ENOSPC)
        self.assertEqual(io.file_receipt(self.local), original)
        self.assertEqual(list(self.local.parent.glob("*.partial")), [unrelated])

    def test_local_corrupt_readback_cannot_replace_prior_checkpoint(self):
        original = io.atomic_checkpoint(self.local, self.payload)
        broken = {**self.payload, "phase_step": 999}
        with patch.object(io.torch, "load", return_value=broken):
            with self.assertRaisesRegex(ValueError, "phase_step"):
                io.atomic_checkpoint(self.local, self.payload)
        self.assertEqual(io.file_receipt(self.local), original)

    def test_publish_success_idempotent_and_immutable(self):
        local_receipt = io.atomic_checkpoint(self.local, self.payload)
        owner = Mock()
        receipt = io.publish_file(self.local, self.remote, owner)
        self.assertEqual(receipt["sha256"], local_receipt["sha256"])
        self.assertEqual(receipt["bytes"], local_receipt["bytes"])
        self.assertEqual(receipt, io.publish_file(self.local, self.remote, owner))
        io.atomic_checkpoint(self.local, {**self.payload, "phase_step": 64})
        with self.assertRaises(io.DurablePublishError) as failure:
            io.publish_file(self.local, self.remote, owner)
        self.assertEqual(failure.exception.local_path, str(self.local))
        self.assertEqual(io.file_receipt(self.remote), receipt)
        self.assertEqual(torch.load(self.local, weights_only=True)["phase_step"], 64)

    def test_remote_disk_full_keeps_local_and_previous_version(self):
        io.atomic_checkpoint(self.local, self.payload)
        prior = io.publish_file(self.local, self.remote, lambda: None)
        new_target = self.remote.with_name("step_64.pt")
        io.atomic_checkpoint(self.local, {**self.payload, "phase_step": 64})
        local_receipt = io.file_receipt(self.local)
        with patch.object(io.os, "fsync", side_effect=OSError(errno.ENOSPC, "remote quota")) as fsync:
            with self.assertRaises(io.DurablePublishError) as error:
                io.publish_file(self.local, new_target, lambda: None)
        self.assertEqual(fsync.call_count, 2)
        self.assertEqual(error.exception.local_path, str(self.local))
        self.assertEqual(io.file_receipt(self.local), local_receipt)
        self.assertEqual(io.file_receipt(self.remote), prior)
        self.assertFalse(new_target.exists())
        self.assertFalse(list(new_target.parent.glob("*.partial")))

    def test_remote_readback_corruption_is_never_promoted(self):
        io.atomic_checkpoint(self.local, self.payload)
        actual_receipt = io.file_receipt

        def corrupted(path):
            result = actual_receipt(path)
            if Path(path).suffix == ".partial":
                result["sha256"] = "0" * 64
            return result

        with patch.object(io, "file_receipt", side_effect=corrupted):
            with self.assertRaises(io.DurablePublishError):
                io.publish_file(self.local, self.remote, lambda: None)
        self.assertTrue(self.local.is_file())
        self.assertFalse(self.remote.exists())

    def test_transient_remote_error_retries_once_without_reserializing(self):
        local = io.atomic_checkpoint(self.local, self.payload)
        real_fsync = io.os.fsync
        calls = []

        def once(fd):
            calls.append(fd)
            if len(calls) == 1:
                raise OSError(errno.EIO, "temporary transport failure")
            return real_fsync(fd)

        with patch.object(io.os, "fsync", side_effect=once), patch.object(io.torch, "save") as save:
            receipt = io.publish_file(self.local, self.remote, lambda: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(receipt["sha256"], local["sha256"])
        save.assert_not_called()

    def test_owner_loss_propagates_and_never_promotes(self):
        io.atomic_checkpoint(self.local, self.payload)
        owner = Mock(side_effect=[None, RuntimeError("ownership lost")])
        with self.assertRaisesRegex(RuntimeError, "ownership lost"):
            io.publish_file(self.local, self.remote, owner)
        self.assertEqual(owner.call_count, 2)
        self.assertFalse(self.remote.exists())
        self.assertTrue(self.local.exists())


if __name__ == "__main__":
    unittest.main()

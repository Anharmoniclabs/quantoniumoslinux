"""Fault-injected Drive API transport tests. No Google account or network used."""

import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from types import SimpleNamespace
import unittest

from phistate_drive_transport import DriveStore, DriveTransferError, CHUNK, RETRIES, hashes


class Request:
    def __init__(self, action): self.action = action
    def execute(self, num_retries=0): return self.action()
    def next_chunk(self, num_retries=0):
        assert num_retries == RETRIES
        return None, self.action()


class Media:
    def __init__(self, filename, *, mimetype, chunksize, resumable):
        assert chunksize == CHUNK and resumable
        self.path = filename


class Download:
    def __init__(self, stream, request, chunksize):
        assert chunksize == CHUNK
        self.stream, self.request = stream, request
    def next_chunk(self, num_retries=0):
        assert num_retries == RETRIES
        self.stream.write(self.request.execute())
        return None, True


class FakeDrive:
    def __init__(self):
        self.records, self.data, self.deleted = {}, {}, []
        self.counter, self.bad_md5, self.fail_pointer = 0, False, False
        self.fail_roles = set()
        self.quota = {"limit": str(2**40), "usage": "0"}
        self.list_result = []
        self.upload_count = 0

    def files(self): return self
    def about(self): return self

    def get(self, fileId=None, **kwargs):
        return Request(lambda: {"storageQuota": dict(self.quota)} if fileId is None else copy.deepcopy(self.records[fileId]))

    def get_media(self, fileId, **kwargs): return Request(lambda: self.data[fileId])

    def list(self, **kwargs):
        return Request(lambda: {"files": copy.deepcopy(self.list_result)})

    def create(self, body, media_body=None, **kwargs):
        def create():
            self.counter += 1
            key = f"file_{self.counter}"
            self.records[key] = {"id": key, **copy.deepcopy(body)}
            self.records[key]["modifiedTime"] = f"{self.counter:012d}"
            self._media(key, media_body)
            return copy.deepcopy(self.records[key])
        return Request(create)

    def _media(self, key, media):
        if media is not None:
            self.upload_count += 1
            data = Path(media.path).read_bytes()
            self.data[key] = data
            self.records[key].update(size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
            if self.bad_md5:
                self.records[key]["md5Checksum"] = "wrong"
                self.bad_md5 = False

    def update(self, fileId, body, media_body, **kwargs):
        def update():
            role = body["appProperties"]["phistate_role"]
            if (self.fail_pointer and role == "checkpoint_pointer") or role in self.fail_roles:
                raise OSError("connection lost")
            self.records[fileId].update(copy.deepcopy(body))
            self._media(fileId, media_body)
            return copy.deepcopy(self.records[fileId])
        return Request(update)

    def delete(self, fileId, **kwargs):
        def delete():
            self.deleted.append(fileId)
            del self.records[fileId]
            self.data.pop(fileId, None)
            return {}
        return Request(delete)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.api = FakeDrive()
        self.store = DriveStore(self.api, media_file_factory=Media, media_download_factory=Download)
        self.local = self.root / "latest.pt"
        self.local.write_bytes(b"complete local checkpoint")

    def tearDown(self): self.tmp.cleanup()

    def test_known_quota_rejects_before_upload_unknown_limit_allowed(self):
        self.api.quota = {"limit": "100", "usage": "99"}
        with self.assertRaisesRegex(DriveTransferError, "free bytes"):
            self.store.upload(self.local, "parent", "model.pt", "run_token")
        self.assertEqual(self.api.upload_count, 0)
        self.assertTrue(self.local.is_file())
        self.api.quota = {"usage": "1000000"}
        self.assertFalse(self.store.quota_guard(2**50)["quota_known"])

    def test_download_hash_required_and_existing_file_preserved_on_corruption(self):
        self.api.data["source"] = b"correct archive"
        expected = hashlib.sha256(b"correct archive").hexdigest()
        target = self.root / "source.pt"
        target.write_bytes(b"existing bytes")
        self.api.data["source"] = b"damaged response"
        with self.assertRaises(DriveTransferError): self.store.download("source", target, expected)
        self.assertEqual(target.read_bytes(), b"existing bytes")
        self.assertEqual(list(self.root.glob(".*.partial")), [])
        self.api.data["source"] = b"correct archive"
        receipt = self.store.download("source", target, expected)
        self.assertEqual(receipt["sha256"], expected)
        self.assertEqual(target.read_bytes(), b"correct archive")

    def test_upload_verified_against_cloud_metadata(self):
        result = self.store.upload(self.local, "parent", "model.pt", "run_token")
        self.assertEqual(result["sha256"], hashlib.sha256(self.local.read_bytes()).hexdigest())
        self.assertEqual(result["md5"], self.api.records[result["id"]]["md5Checksum"])
        self.assertEqual(result["size"], self.local.stat().st_size)

    def test_bad_checkpoint_md5_never_promotes_pointer(self):
        self.api.bad_md5 = True
        with self.assertRaisesRegex(DriveTransferError, "local file remains"):
            self.store.sync_checkpoint(self.local, "parent", "run_token", 32)
        self.assertFalse(any(r["name"] == "latest.json" for r in self.api.records.values()))
        self.assertEqual(self.local.read_bytes(), b"complete local checkpoint")
        self.assertEqual(self.api.deleted, [])

    def test_failed_pointer_keeps_prior_snapshots_and_local(self):
        state = self.store.sync_checkpoint(self.local, "parent", "run_token", 32)
        previous_id, pointer_id = state["snapshots"][-1]["id"], state["pointer_id"]
        old_pointer = self.api.data[pointer_id]
        self.local.write_bytes(b"step 128")
        self.api.fail_pointer = True
        with self.assertRaises(DriveTransferError):
            self.store.sync_checkpoint(self.local, "parent", "run_token", 128, state)
        self.assertIn(previous_id, self.api.records)
        self.assertEqual(self.api.data[pointer_id], old_pointer)
        self.assertEqual(self.local.read_bytes(), b"step 128")
        self.assertEqual(self.api.deleted, [])

    def test_retention_deletes_only_verified_owned_superseded_snapshot(self):
        state, snapshots = None, []
        for step in (32, 128, 256):
            self.local.write_bytes(f"checkpoint {step}".encode())
            state = self.store.sync_checkpoint(self.local, "parent", "run_token", step, state)
            snapshots.append(state["snapshots"][-1]["id"])
        self.assertEqual(self.api.deleted, snapshots[:1])
        self.assertEqual([s["step"] for s in state["snapshots"]], [128, 256])
        pointer = json.loads(self.api.data[state["pointer_id"]])
        self.assertEqual(pointer["latest"]["step"], 256)
        self.assertEqual(pointer["previous"]["step"], 128)

    def test_imported_snapshot_is_never_deleted(self):
        state = self.store.sync_checkpoint(self.local, "parent", "run_token", 32)
        imported = state["snapshots"][-1]["id"]
        self.store._created.remove(imported)  # represents restored state from another process
        for step in (128, 256):
            self.local.write_bytes(str(step).encode())
            state = self.store.sync_checkpoint(self.local, "parent", "run_token", step, state)
        self.assertIn(imported, self.api.records)
        self.assertNotIn(imported, self.api.deleted)

    def test_deferred_cleanup_preserves_last_committed_index_on_failure(self):
        state, snapshots = None, []
        for step in (32, 128, 256):
            self.local.write_bytes(f"checkpoint {step}".encode())
            state = self.store.sync_checkpoint(self.local, "parent", "run_token", step, state, defer_cleanup=True)
            snapshots.append(state["snapshots"][-1]["id"])
        # Aggregate index still points to the first checkpoint because its two
        # later uploads failed. No snapshot may disappear before index commit.
        self.assertEqual(self.api.deleted, [])
        self.assertIn(snapshots[0], self.api.records)
        self.assertEqual([r["id"] for r in state["pending_cleanup"]], snapshots[:1])
        protected = self.store.cleanup_snapshots("parent", "run_token", state, protected_ids=[snapshots[0]])
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(len(protected["pending_cleanup"]), 1)
        # Only after the aggregate pointer commits the newest checkpoint can
        # the obsolete first snapshot be removed.
        committed = self.store.cleanup_snapshots("parent", "run_token", protected, protected_ids=[snapshots[-1]])
        self.assertEqual(self.api.deleted, snapshots[:1])
        self.assertEqual(committed["pending_cleanup"], [])

    def test_restart_never_deletes_imported_pending_snapshots(self):
        state = None
        for step in (32, 128, 256):
            self.local.write_bytes(str(step).encode())
            state = self.store.sync_checkpoint(self.local, "parent", "run_token", step, state, defer_cleanup=True)
        restarted = DriveStore(self.api, media_file_factory=Media, media_download_factory=Download)
        result = restarted.cleanup_snapshots("parent", "run_token", state)
        self.assertEqual(len(result["pending_cleanup"]), 1)
        self.assertEqual(self.api.deleted, [])

    def test_repeated_sync_revalidates_prior_cloud_snapshot(self):
        state = self.store.sync_checkpoint(self.local, "parent", "run_token", 32)
        uploads = self.api.upload_count
        self.assertEqual(self.store.sync_checkpoint(self.local, "parent", "run_token", 32, state), state)
        self.assertEqual(self.api.upload_count, uploads)
        self.api.records[state["snapshots"][-1]["id"]]["md5Checksum"] = "changed"
        with self.assertRaisesRegex(DriveTransferError, "snapshot changed"):
            self.store.sync_checkpoint(self.local, "parent", "run_token", 32, state)

    def test_unowned_pointer_cannot_be_overwritten(self):
        previous = self.store.upload_json({"hello": 1}, "parent", "latest.json", "another_run", role="checkpoint_pointer")
        with self.assertRaises(DriveTransferError):
            self.store.upload_json({"hello": 2}, "parent", "latest.json", "run_token",
                                   previous["id"], role="checkpoint_pointer")
        self.assertEqual(json.loads(self.api.data[previous["id"]]), {"hello": 1})

    def test_folder_reuse_requires_exact_owned_match(self):
        folder = self.store.ensure_folder("parent", "arm", "run_token")
        self.api.list_result = [self.api.records[folder]]
        self.assertEqual(self.store.ensure_folder("parent", "arm", "run_token"), folder)
        with self.assertRaises(DriveTransferError): self.store.ensure_folder("parent", "arm", "other_run")
        self.api.list_result *= 2
        with self.assertRaises(DriveTransferError): self.store.ensure_folder("parent", "arm", "run_token")

    def test_wrapper_restart_after_two_failed_index_commits(self):
        """Exercise real Backups/newest_index code without importing torch/CUDA."""
        source = Path(__file__).with_name("phistate_a100_storage_recovery.py")
        if not source.is_file(): self.skipTest("Storage recovery coordinator is unavailable")
        needed = {"safe_relative", "_index_files", "newest_index", "Backups"}
        tree = ast.Module(body=[n for n in ast.parse(source.read_text()).body
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in needed], type_ignores=[])
        def atomic_json(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        scope = {"__file__": str(source), "Path": Path, "PurePosixPath": PurePosixPath,
                 "json": json, "hashlib": hashlib, "file_receipt": hashes, "atomic_json": atomic_json,
                 "journal": SimpleNamespace(OUTPUT="phistate_a100_base_v1", SEEDS=(20261101, 20261102)),
                 "FORMAT": "test-recovery", "RECOVERY": {"root_folder_id": "parent", "old_experiment_request_key": "test-key"}}
        exec(compile(tree, str(source), "exec"), scope)
        original_list = self.api.list
        def listing(**kwargs):
            if "recovery_index" in kwargs.get("q", ""):
                return Request(lambda: {"files": [copy.deepcopy(r) for r in self.api.records.values()
                    if r.get("appProperties", {}).get("phistate_role") == "recovery_index"]})
            return original_list(**kwargs)
        self.api.list = listing
        work = self.root / "work"
        path = work / "project/phistate_a100_base_v1/dense_20261101/latest.pt"
        path.parent.mkdir(parents=True)
        state = {"format": "test-recovery", "experiment": "test-key", "token": "run_token",
                 "folder_id": "run_folder", "folders": {}, "arms": {}, "files": {}, "steps": {}}
        backups = scope["Backups"](self.store, work, state)
        path.write_bytes(b"checkpoint 32")
        backups.weights(path, 32)
        backups.commit_index("training")
        committed = state["files"]["phistate_a100_base_v1/dense_20261101/latest.pt"]["id"]
        self.api.fail_roles.add("recovery_index")
        for step in (128, 256):
            path.write_bytes(f"checkpoint {step}".encode())
            backups.weights(path, step)
            with self.assertRaises(DriveTransferError): backups.commit_index("training")
        self.assertIn(committed, self.api.records)
        old_index = scope["newest_index"](self.store)
        self.assertEqual(old_index["steps"], {"dense_20261101": 32})
        # A new Python process reuses the newer local checkpoint, while the
        # previous cloud index remains valid throughout retry.
        self.api.fail_roles.clear()
        restarted_store = DriveStore(self.api, media_file_factory=Media, media_download_factory=Download)
        restarted_state = json.loads((work / "transport.json").read_text())
        restarted = scope["Backups"](restarted_store, work, restarted_state)
        restarted.weights(path, 256)
        restarted.commit_index("training")
        self.assertEqual(path.read_bytes(), b"checkpoint 256")
        current_index = scope["newest_index"](restarted_store)
        self.assertEqual(current_index["steps"], {"dense_20261101": 256})
        checkpoint = current_index["files"]["phistate_a100_base_v1/dense_20261101/latest.pt"]
        self.assertEqual(self.api.data[checkpoint["id"]], b"checkpoint 256")


if __name__ == "__main__": unittest.main()

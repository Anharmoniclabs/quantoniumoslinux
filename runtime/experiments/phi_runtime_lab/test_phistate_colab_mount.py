"""Exercise Colab startup recovery without a GPU or a real Google session."""

import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import build_phistate_a100_base_notebook as builder


RELATIVE_PROJECT = Path("Quantonium/PhiState-L4/whitespace_repair_v1")


class TestColabDriveMount(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.mountpoint = self.directory / "drive"

        # Keep launch() uncalled and avoid evaluating its embedded package.
        parsed = ast.parse(builder.LAUNCHER)
        definitions = [node for node in parsed.body
                       if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        self.ns = {"__name__": "mount_test_launcher"}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "launcher", "exec"), self.ns)
        self.ns.update(MOUNT=self.mountpoint,
                       ROOT=self.mountpoint / "MyDrive" / RELATIVE_PROJECT,
                       REPORT=self.mountpoint / "MyDrive" / RELATIVE_PROJECT /
                       "phistate_a100_base_v1/report.json")
        fingerprint = next(node.value.value for node in parsed.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name)
                                   and target.id == "CORPUS_FINGERPRINT"
                                   for target in node.targets))
        self.ns["CORPUS_FINGERPRINT"] = fingerprint
        self.parse_mounts = self.ns["drive_mounts"]
        self.mounts = Mock(return_value=[])
        self.ns["drive_mounts"] = self.mounts

        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        drive = types.ModuleType("google.colab.drive")
        self.mount = Mock()
        drive.mount = self.mount
        colab.drive = drive
        google.colab = colab
        modules = patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                           "google.colab.drive": drive})
        modules.start()
        self.addCleanup(modules.stop)

    def project(self, mountpoint, home="MyDrive"):
        project = mountpoint / home / RELATIVE_PROJECT
        manifest = project / "corpus_repaired_v1/manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"format": "PhiMoECorpus-v1",
                                        "data_fingerprint": self.ns["CORPUS_FINGERPRINT"]}),
                            encoding="utf-8")
        return project

    def succeeds(self, mountpoint, **kwargs):
        self.project(Path(mountpoint))
        self.mounts.return_value = [Path(mountpoint)]

    def test_mountinfo_decodes_spaces_and_filters_other_filesystems(self):
        records = (
            "35 25 0:1 / / rw - overlay overlay rw\n"
            "36 25 0:42 / /content/drive rw - fuse.drive drive rw\n"
            "37 25 0:43 / /content/My\\040Drive rw - fuse.drive drive rw\n"
            "38 25 0:44 / /content/other rw - fuse.sshfs user@example rw\n"
        )
        with patch.object(Path, "read_text", return_value=records):
            self.assertEqual(self.parse_mounts(),
                             [Path("/content/drive"), Path("/content/My Drive")])

    def test_reuses_expected_mount_without_remounting(self):
        project = self.project(self.mountpoint)
        self.mounts.return_value = [self.mountpoint]
        self.ns["ensure_drive"]()
        self.mount.assert_not_called()
        self.assertEqual(self.ns["ROOT"], project)

    def test_reuses_other_mount_and_updates_report_path(self):
        other = self.directory / "shared_mount"
        project = self.project(other, home="My Drive")
        self.mounts.return_value = [other]
        self.ns["ensure_drive"]()
        self.mount.assert_not_called()
        self.assertEqual(self.ns["MOUNT"], other)
        self.assertEqual(self.ns["ROOT"], project)
        self.assertEqual(self.ns["REPORT"], project / "phistate_a100_base_v1/report.json")

    def test_wrong_account_does_not_replace_existing_mount(self):
        (self.mountpoint / "MyDrive").mkdir(parents=True)
        self.mounts.return_value = [self.mountpoint]
        with self.assertRaises(RuntimeError):
            self.ns["ensure_drive"]()
        self.mount.assert_not_called()

    def test_transient_failure_is_retried_then_verified(self):
        attempts = []

        def transient(mountpoint, **kwargs):
            attempts.append(mountpoint)
            if len(attempts) == 1:
                raise ValueError("mount failed")
            self.succeeds(mountpoint, **kwargs)

        self.mount.side_effect = transient
        self.ns["ensure_drive"]()
        self.assertEqual(self.mount.call_count, 2)
        self.assertEqual(self.ns["ROOT"], self.mountpoint / "MyDrive" / RELATIVE_PROJECT)

    def test_persistent_failure_is_bounded(self):
        self.mount.side_effect = ValueError("mount failed")
        with self.assertRaises(RuntimeError):
            self.ns["ensure_drive"]()
        self.assertEqual(self.mount.call_count, 2)

    def test_nonempty_local_directory_is_preserved(self):
        self.mountpoint.mkdir()
        sentinel = self.mountpoint / "unrelated_job.txt"
        sentinel.write_text("keep this", encoding="utf-8")
        with self.assertRaises((RuntimeError, ValueError, FileExistsError)):
            self.ns["ensure_drive"]()
        self.mount.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this")

    def test_keyboard_interrupt_is_not_retried(self):
        self.mount.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.ns["ensure_drive"]()
        self.assertEqual(self.mount.call_count, 1)

    def test_other_authentication_error_is_not_retried(self):
        class AuthenticationError(RuntimeError):
            pass

        error = AuthenticationError("authorization denied")
        self.mount.side_effect = error
        with self.assertRaises(AuthenticationError) as raised:
            self.ns["ensure_drive"]()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.mount.call_count, 1)


if __name__ == "__main__":
    unittest.main()

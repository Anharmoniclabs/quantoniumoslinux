"""Verified, resumable Drive API transfers; training files remain on local disk.

No mounted Drive paths are used. Upload failures preserve local files and all
previous verified snapshots. Only snapshots created by this Store instance may
be deleted, after a newer checkpoint and pointer have been verified.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

CHUNK = 8 * 1024 * 1024
RETRIES = 3
FIELDS = "id,name,mimeType,size,md5Checksum,parents,appProperties,trashed"
FOLDER = "application/vnd.google-apps.folder"


class DriveTransferError(RuntimeError):
    pass


def hashes(path):
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            sha.update(block); md5.update(block); size += len(block)
    return {"size": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def _token(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value):
        raise ValueError(f"Invalid {label}")
    return value


def _name(value):
    if not isinstance(value, str) or not value or "/" in value or "\x00" in value:
        raise ValueError("Expected a single Drive file name")
    return value


def _quoted(value):
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class DriveStore:
    def __init__(self, service, *, media_file_factory=None, media_download_factory=None):
        self.service = service
        self._media_file = media_file_factory
        self._media_download = media_download_factory
        self._created = set()

    def _factories(self):
        if self._media_file is None or self._media_download is None:
            from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
            self._media_file = self._media_file or MediaFileUpload
            self._media_download = self._media_download or MediaIoBaseDownload

    def metadata(self, file_id):
        return self.service.files().get(fileId=_token(file_id, "file ID"), fields=FIELDS,
                                       supportsAllDrives=True).execute(num_retries=RETRIES)

    def quota_guard(self, min_free_bytes):
        """Only a returned finite quota can establish that available space is low."""
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise ValueError("min_free_bytes must be a nonnegative integer")
        quota = self.service.about().get(fields="storageQuota").execute(num_retries=RETRIES).get("storageQuota", {})
        free = None
        if "limit" in quota and "usage" in quota:
            free = max(0, int(quota["limit"]) - int(quota["usage"]))
            if free < min_free_bytes:
                raise DriveTransferError(
                    f"Drive reports {free:,} free bytes; this upload needs {min_free_bytes:,}. "
                    "The local checkpoint is retained. Free Drive space before retrying.")
        return {"required_bytes": min_free_bytes, "free_bytes": free, "quota_known": free is not None}

    def download(self, file_id, destination, expected_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("A verified SHA-256 is required for every source download")
        destination = Path(destination)
        if destination.is_file() and hashes(destination)["sha256"] == expected_sha256:
            return {"id": file_id, "path": str(destination), **hashes(destination)}
        self._factories()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
        temporary = Path(temporary)
        try:
            request = self.service.files().get_media(fileId=_token(file_id, "file ID"), supportsAllDrives=True)
            with os.fdopen(fd, "wb") as stream:
                transfer = self._media_download(stream, request, chunksize=CHUNK)
                done = False
                while not done:
                    _, done = transfer.next_chunk(num_retries=RETRIES)
                stream.flush(); os.fsync(stream.fileno())
            receipt = hashes(temporary)
            if receipt["sha256"] != expected_sha256:
                raise DriveTransferError("Source download SHA-256 mismatch")
            os.replace(temporary, destination)
            return {"id": file_id, "path": str(destination), **receipt}
        except Exception as exc:
            raise DriveTransferError(f"Drive download failed for {destination}; existing local file retained ({type(exc).__name__}).") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _owned(self, record, parent_id, run_token, role):
        return (not record.get("trashed") and parent_id in record.get("parents", [])
                and record.get("appProperties", {}).get("phistate_run") == run_token
                and record.get("appProperties", {}).get("phistate_role") == role)

    def ensure_folder(self, parent_id, name, run_token):
        _token(parent_id, "parent ID"); _token(run_token, "run token"); _name(name)
        query = f"{_quoted(parent_id)} in parents and name = {_quoted(name)} and trashed = false"
        matches, page = [], None
        while True:
            result = self.service.files().list(q=query, spaces="drive", pageSize=100,
                fields=f"nextPageToken,files({FIELDS})", pageToken=page).execute(num_retries=RETRIES)
            matches.extend(result.get("files", [])); page = result.get("nextPageToken")
            if not page: break
        if matches:
            if len(matches) != 1 or matches[0].get("mimeType") != FOLDER or not self._owned(matches[0], parent_id, run_token, "folder"):
                raise DriveTransferError(f"Existing Drive folder name is ambiguous or belongs to another run: {name}")
            return matches[0]["id"]
        record = self.service.files().create(body={"name": name, "parents": [parent_id], "mimeType": FOLDER,
            "appProperties": {"phistate_run": run_token, "phistate_role": "folder"}}, fields=FIELDS,
            supportsAllDrives=True).execute(num_retries=0)
        # Folder creation is not retried blindly: a lost response may have created it.
        verified = self.metadata(record["id"])
        if verified.get("mimeType") != FOLDER or not self._owned(verified, parent_id, run_token, "folder"):
            raise DriveTransferError("Created Drive folder failed ownership verification")
        self._created.add(verified["id"])
        return verified["id"]

    def _upload(self, local, parent_id, name, run_token, role, file_id=None):
        local = Path(local)
        _token(parent_id, "parent ID"); _token(run_token, "run token"); _token(role, "role"); _name(name)
        self._factories()
        initial = hashes(local)
        self.quota_guard(initial["size"] + 1024 * 1024)
        try:
            body = {"name": name, "appProperties": {"phistate_run": run_token, "phistate_role": role}}
            media = self._media_file(str(local), mimetype="application/octet-stream", chunksize=CHUNK, resumable=True)
            if file_id is None:
                body["parents"] = [parent_id]
                request = self.service.files().create(body=body, media_body=media, fields="id", supportsAllDrives=True)
            else:
                existing = self.metadata(file_id)
                if existing.get("name") != name or not self._owned(existing, parent_id, run_token, role):
                    raise DriveTransferError("Refusing to update an unowned Drive object")
                request = self.service.files().update(fileId=file_id, body=body, media_body=media,
                                                      fields="id", supportsAllDrives=True)
            result = None
            # googleapiclient bounds transient retries within each chunk request.
            for _ in range(max(1, math.ceil(initial["size"] / CHUNK)) + 8):
                _, result = request.next_chunk(num_retries=RETRIES)
                if result is not None: break
            if result is None:
                raise DriveTransferError("Resumable upload made insufficient progress")
            remote_id = result["id"]
            if file_id is None: self._created.add(remote_id)
            remote = self.metadata(remote_id)
            if (int(remote.get("size", -1)) != initial["size"] or remote.get("md5Checksum") != initial["md5"]
                    or not self._owned(remote, parent_id, run_token, role) or hashes(local) != initial):
                raise DriveTransferError("Uploaded object failed size, checksum, ownership or local-stability verification")
            return {"id": remote_id, "name": name, "parent_id": parent_id, "run_token": run_token,
                    "role": role, "local_path": str(local), **initial}
        except Exception as exc:
            raise DriveTransferError(f"Drive upload failed; local file remains at {local} ({type(exc).__name__}).") from exc

    def upload(self, local, parent_id, name, run_token, role="artifact"):
        return self._upload(local, parent_id, name, run_token, role)

    def upload_json(self, value, parent_id, name, run_token, file_id=None, role="metadata"):
        with tempfile.TemporaryDirectory(prefix="phistate_drive_json_") as tmp:
            path = Path(tmp) / "record.json"
            path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            result = self._upload(path, parent_id, name, run_token, role, file_id)
            result.pop("local_path", None)
            return result

    def sync_checkpoint(self, local, parent_id, run_token, step, state=None, *, defer_cleanup=False):
        """Publish immutable weights then latest.json. Keep the last two verified snapshots."""
        if type(step) is not int or step < 0: raise ValueError("Invalid checkpoint step")
        state = dict(state or {})
        previous = list(state.get("snapshots", []))
        identity = hashes(local)
        if previous and previous[-1]["sha256"] == identity["sha256"] and previous[-1]["step"] == step:
            remote = self.metadata(previous[-1]["id"])
            if (not self._owned(remote, parent_id, run_token, "checkpoint")
                    or int(remote.get("size", -1)) != identity["size"] or remote.get("md5Checksum") != identity["md5"]):
                raise DriveTransferError("Previously verified Drive snapshot changed; local checkpoint retained")
            return state
        if previous and step <= previous[-1]["step"]:
            raise DriveTransferError("Checkpoint steps must advance monotonically")
        snapshot = self.upload(local, parent_id, f"step_{step:09d}_{identity['sha256'][:16]}.pt", run_token, role="checkpoint")
        snapshot["step"] = step
        snapshots = previous + [snapshot]
        pointer = {"format": "PhiState-Drive-checkpoint-v1", "run_token": run_token,
                   "latest": snapshot, "previous": snapshots[-2] if len(snapshots) > 1 else None}
        receipt = self.upload_json(pointer, parent_id, "latest.json", run_token,
                                   file_id=state.get("pointer_id"), role="checkpoint_pointer")
        pending = {r["id"]: r for r in state.get("pending_cleanup", []) + snapshots[:-2]}
        result = {"pointer_id": receipt["id"], "snapshots": snapshots[-2:],
                  "pending_cleanup": list(pending.values()), "cleanup_errors": []}
        if not defer_cleanup:
            result = self.cleanup_snapshots(parent_id, run_token, result)
        return result

    def cleanup_snapshots(self, parent_id, run_token, state, protected_ids=()):
        """Prune only after the caller's aggregate recovery index is committed.

        The newest two snapshots and caller-protected file IDs are retained.
        Unknown creation IDs are also retained, including snapshots from a
        previous process; a fresh runtime never deletes imported recovery data.
        """
        result = dict(state)
        result["pending_cleanup"], result["cleanup_errors"] = [], []
        protected = set(protected_ids) | {r["id"] for r in state.get("snapshots", [])}
        for old in state.get("pending_cleanup", []):
            # The current object metadata and our in-memory creation ledger must
            # both authorize retention cleanup. Never delete imported checkpoints.
            if old["id"] not in self._created or old["id"] in protected:
                result["pending_cleanup"].append(old)
                continue
            try:
                remote = self.metadata(old["id"])
                if (not self._owned(remote, parent_id, run_token, "checkpoint")
                        or int(remote.get("size", -1)) != old["size"] or remote.get("md5Checksum") != old["md5"]):
                    raise DriveTransferError("Superseded snapshot ownership changed; retained")
                self.service.files().delete(fileId=old["id"], supportsAllDrives=True).execute(num_retries=RETRIES)
                self._created.remove(old["id"])
            except Exception as exc:
                result["pending_cleanup"].append(old)
                result["cleanup_errors"].append({"id": old["id"], "error_type": type(exc).__name__})
        return result

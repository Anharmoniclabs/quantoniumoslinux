"""Validated local checkpoints and immutable, verified copies to remote storage.

The training checkpoint must be on the runtime's local disk. Publishing is a
separate operation: it never invokes torch.save on the remote filesystem. Use a
new, versioned destination for each publication, and update a small run pointer
only after publish_file returns. SHA verification through a mounted filesystem
verifies the readable copy, not the storage provider's server-side durability.
"""

from collections import OrderedDict
import errno
import hashlib
import math
import os
from pathlib import Path
import tempfile
import uuid

import torch


CHUNK_BYTES = 4 * 1024 * 1024


class DurablePublishError(RuntimeError):
    """Remote publication failed; the complete local checkpoint is retained."""

    def __init__(self, local_path, destination, cause):
        self.local_path = str(local_path)
        self.destination = str(destination)
        self.cause = cause
        super().__init__(
            f"Remote checkpoint publication failed: {type(cause).__name__}: {cause}. "
            f"Complete local recovery file retained at {self.local_path}. "
            "Keep this Colab runtime connected until the file is backed up."
        )


def file_receipt(path):
    """Read every byte and return its size and SHA-256."""
    path = Path(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            count += len(chunk)
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


def _same_bytes(left, right):
    return all(left[k] == right[k] for k in ("bytes", "sha256"))


def _cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, OrderedDict):
        result = OrderedDict((k, _cpu_snapshot(v)) for k, v in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = _cpu_snapshot(value._metadata)
        return result
    if type(value) is dict:
        return {k: _cpu_snapshot(v) for k, v in value.items()}
    if type(value) is list:
        return [_cpu_snapshot(v) for v in value]
    if type(value) is tuple:
        return tuple(_cpu_snapshot(v) for v in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool, torch.dtype, torch.device)):
        return value
    raise TypeError(f"Unsupported checkpoint value type: {type(value).__name__}")


def _validate_roundtrip(expected, actual, where="checkpoint"):
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.device.type != "cpu"
                or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(expected, actual)):
            raise ValueError(f"Checkpoint tensor readback mismatch at {where}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Checkpoint mapping readback mismatch at {where}")
        for key in expected:
            _validate_roundtrip(expected[key], actual[key], f"{where}.{key}")
        if hasattr(expected, "_metadata"):
            _validate_roundtrip(expected._metadata, getattr(actual, "_metadata", None), f"{where}._metadata")
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(expected) != len(actual):
            raise ValueError(f"Checkpoint sequence readback mismatch at {where}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _validate_roundtrip(left, right, f"{where}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        if not isinstance(actual, float) or not math.isnan(actual):
            raise ValueError(f"Checkpoint scalar readback mismatch at {where}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"Checkpoint scalar readback mismatch at {where}")


def _remove_own_partial(path):
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # The remote filesystem may be unavailable. Never replace the
            # primary exception with a cleanup failure or remove other files.
            pass


def atomic_checkpoint(path, payload):
    """Save and validate a CPU snapshot on local disk, then replace its target.

    A failed serialization or readback leaves the previous target untouched.
    The parent directory must be a local runtime directory, not mounted Drive.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError(f"Checkpoint target must not be a symlink: {target}")
    snapshot = _cpu_snapshot(payload)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
        before = file_receipt(temporary)
        restored = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_roundtrip(snapshot, restored)
        del restored
        after = file_receipt(temporary)
        if not _same_bytes(before, after):
            raise ValueError("Local checkpoint changed during its readback verification")
        os.replace(temporary, target)
        return {**after, "path": str(target)}
    finally:
        _remove_own_partial(temporary)


def publish_file(local, destination, ownership_check, *, attempts=2):
    """Copy a local file to a new remote destination and verify every byte.

    Destinations are immutable. An existing identical file is idempotent; an
    existing different file is never replaced. Older checkpoint versions stay
    in place even when a later publication fails. Callers should choose a
    versioned filename containing the local SHA-256 or optimizer step.
    """
    source, target = Path(local), Path(destination)
    if attempts not in (1, 2):
        raise ValueError("Publication attempts must be one or two")
    if source.resolve() == target.resolve():
        raise ValueError("Local recovery file and remote publication must be distinct")
    expected = file_receipt(source)
    last_error = None
    for _ in range(attempts):
        partial = None
        # An ownership failure is not a transient storage error and must never
        # be retried as if this worker still had permission to publish.
        ownership_check()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                raise FileExistsError(errno.EEXIST, "Refusing linked publication target", str(target))
            if target.exists():
                existing = file_receipt(target)
                if not _same_bytes(expected, existing):
                    raise FileExistsError(errno.EEXIST, "Publication destination is immutable", str(target))
                ownership_check()
                return existing
            partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
            copied = hashlib.sha256()
            copied_bytes = 0
            with source.open("rb") as reader, partial.open("xb") as writer:
                for chunk in iter(lambda: reader.read(CHUNK_BYTES), b""):
                    if writer.write(chunk) != len(chunk):
                        raise OSError(errno.EIO, "Short checkpoint write")
                    copied.update(chunk)
                    copied_bytes += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            copied_receipt = {"bytes": copied_bytes, "sha256": copied.hexdigest()}
            if not _same_bytes(expected, copied_receipt):
                raise OSError(errno.EIO, "Local recovery file changed while publishing")
            verified = file_receipt(partial)
            if not _same_bytes(expected, verified):
                raise OSError(errno.EIO, "Remote checkpoint readback mismatch")
            ownership_check()
            # One run owner serializes publishers. Recheck before promotion to
            # preserve a preexisting target that appeared during copying.
            if target.exists() or target.is_symlink():
                raise FileExistsError(errno.EEXIST, "Publication target appeared during copying", str(target))
            os.replace(partial, target)
            final = file_receipt(target)
            if not _same_bytes(expected, final):
                raise OSError(errno.EIO, "Published checkpoint readback mismatch")
            return final
        except OSError as exc:
            last_error = exc
            if isinstance(exc, FileExistsError):
                break
        finally:
            _remove_own_partial(partial)
    raise DurablePublishError(source, target, last_error) from last_error

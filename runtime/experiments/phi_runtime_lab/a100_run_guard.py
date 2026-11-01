"""One writer per A100 experiment, with a persistent owner record beside its output.

The local flock covers this runtime. The shared guard uses exclusive mkdir for
numbered generations. Generations are never removed or reused, so a concurrent
recovery cannot unlink a fresh claim. This requires shared, coherent directory
semantics: keep one live Colab runtime per job. On a disconnected remote host,
recover=True is the operator's confirmation that the previous worker has ended,
not a failure detector. It still requires an expired heartbeat unless the old
process is proved absent on this same host and boot. Call owner.check()
immediately before publishing checkpoint/data files when wrapping other writers.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
import uuid


HEARTBEAT_SECONDS = 20
STALE_SECONDS = 600
_CURRENT = ContextVar('a100_run_owner', default=None)


@contextmanager
def _local_run_lock(path):
    # Same local flock convention as chat_recovery_train.run_lock, using our
    # sibling path so the underlying coordinator can acquire its own run lock.
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f'phistate_chat_{key}.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'A worker on this runtime already owns {path}') from None
        yield


def _host():
    return socket.gethostname(), Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _proc_pid():
    # Use the PID visible in the mounted procfs even in nested PID namespaces.
    return int(Path('/proc/self/stat').read_text().split(' ', 1)[0])


def _process_identity(pid):
    # Linux /proc/PID/stat includes parenthesized command names that may contain
    # spaces or parentheses. starttime is field 22; state is field 3.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return fields[0], fields[19]


def _same_live_process(info):
    if not info:
        return False
    if (info['host'], info['boot_id']) != _host():
        return False
    try:
        status, started = _process_identity(info['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    except PermissionError:
        return True  # Inaccessible on our host is not evidence of exit.
    return status != 'Z' and started == info['process_start']


def _latest(root):
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise RuntimeError(f'Shared guard path is not a real directory: {root}')
        entries = list(root.iterdir())
        if any(not p.name.isascii() or not p.name.isdecimal() or len(p.name) != 10
               or not stat.S_ISDIR(p.lstat().st_mode) for p in entries):
            raise RuntimeError(f'Unexpected contents in shared owner directory: {root}')
    except OSError as exc:
        raise RuntimeError(
            f'Shared guard directory is missing or unreadable: {root} '
            f'({type(exc).__name__}: {exc})') from exc
    return max(entries, default=None)


def _write_once(path, record):
    # Metadata is read concurrently. A partial owner is never interpreted as a
    # release. The unique generation means only its original writer uses this tmp.
    tmp = path.with_suffix('.partial')
    with tmp.open('x', encoding='utf-8') as file:
        json.dump(record, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def _read_owner(gen):
    path = gen/'owner.json'
    if not path.is_file():
        return None  # An interrupted first-ever claim still occupies its slot.
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or not isinstance(data.get('token'), str)
            or not isinstance(data.get('pid'), int) or data['pid'] <= 0
            or not all(isinstance(data.get(k), str) and data[k] for k in
                       ('host', 'boot_id', 'process_start'))):
        raise RuntimeError(f'Invalid owner record, inspect {path}')
    return data


def _allow_next(previous, recover):
    if previous is None:
        return
    info = _read_owner(previous)
    release = previous/'released.json'
    if release.exists():
        if not info or json.loads(release.read_text()).get('token') != info['token']:
            raise RuntimeError(f'Unverified release for {previous}')
        return
    same_host = bool(info and (info['host'], info['boot_id']) == _host())
    live = _same_live_process(info)
    if recover and same_host and not live:
        # /proc proved exit, zombie state, or PID reuse on this exact boot.
        # A remote hostname mismatch alone never permits immediate recovery.
        return
    changed = max(previous.stat().st_mtime, (previous/'owner.json').stat().st_mtime if info else 0)
    age = time.time() - changed
    if age <= STALE_SECONDS or live:
        raise RuntimeError(f'Active or initializing A100 owner at {previous}; do not start another runtime')
    if not recover:
        raise RuntimeError(f'Stale A100 owner at {previous}. Confirm the old runtime has stopped before explicit recovery.')


class _Owner:
    def __init__(self, out, root, generation, token):
        self.out, self.root, self.generation, self.token = out, root, generation, token
        self.error = None
        self.stop = threading.Event()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Cannot verify A100 heartbeat ownership; stopping') from self.error
        info = _read_owner(self.generation)
        if (_latest(self.root) != self.generation or not info or info['token'] != self.token
                or (self.generation/'released.json').exists()):
            raise RuntimeError('This worker no longer holds the latest A100 owner generation')

    def heartbeat(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            try:
                self.check()
                os.utime(self.generation/'owner.json', None)
            except Exception as exc:
                self.error = exc
                return


@contextmanager
def guarded_run(out, recover=False):
    """Acquire a host-local flock and a persistent shared claim. Never clear STOP."""
    out = Path(out).resolve()
    root = out.parent/f'.{out.name}.a100-owner'
    root.parent.mkdir(parents=True, exist_ok=True)
    with _local_run_lock(root):
        root.mkdir(exist_ok=True)
        previous = _latest(root)
        _allow_next(previous, recover)
        generation = root/f'{1 if previous is None else int(previous.name)+1:010d}'
        try:
            generation.mkdir()
        except FileExistsError:
            raise RuntimeError(f'Another runtime claimed {generation}; retry after it has exited') from None
        token = uuid.uuid4().hex
        host, boot = _host()
        pid = _proc_pid()
        _, start = _process_identity(pid)
        _write_once(generation/'owner.json', dict(token=token, host=host, boot_id=boot,
                                                   pid=pid, process_start=start))
        owner = _Owner(out, root, generation, token)
        owner.check()
        active = _CURRENT.set(owner)
        worker = threading.Thread(target=owner.heartbeat, name='a100-owner-heartbeat', daemon=True)
        worker.start()
        primary_error = None
        try:
            yield owner
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                owner.stop.set()
                worker.join()
                owner.check()
                _write_once(generation/'released.json', {'token': token})
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    'A100 owner cleanup also failed; the original failure is preserved. '
                    f'{type(cleanup_error).__name__}: {cleanup_error}')
            finally:
                _CURRENT.reset(active)


def clean_orphans(out):
    """Remove only interrupted first-commit temporary files, under this job's guard.

    Allowlist: request.partial and, inside broad/paired only, latest.pt.partial
    and progress.partial. Retain valid destinations, unrelated files and all
    STOP markers. Coordinator/trainer validation decides whether remaining files
    describe a resumable run. Returns the removed paths.
    """
    out = Path(out).resolve()
    owner = _CURRENT.get()
    if owner is None or owner.out != out:
        raise RuntimeError('clean_orphans requires the matching active guarded_run')
    owner.check()
    candidates = [(out/'request.partial', out/'request.json')]
    for name in ('broad', 'paired'):
        arm = out/name
        if arm.is_symlink():
            raise RuntimeError(f'Unexpected linked arm path {arm}')
        candidates.extend([(arm/'latest.pt.partial', arm/'latest.pt'),
                           (arm/'progress.partial', arm/'progress.json')])
    removed = []
    for partial, destination in candidates:
        if destination.exists() or not partial.exists() and not partial.is_symlink():
            continue
        if partial.is_symlink() or not stat.S_ISREG(partial.stat().st_mode):
            raise RuntimeError(f'Unexpected orphan type: {partial}')
        owner.check()
        partial.unlink()
        removed.append(partial)
    return removed

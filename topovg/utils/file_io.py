"""Atomic persistence and advisory locking for parser/feature writers."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from .backend import LazyModule

torch = LazyModule('torch')

def atomic_json(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def backup_file(path):
    if not path.exists():
        return None
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest = path.with_name(path.name + '.' + stamp + '.bak')
    shutil.copy2(path, dest)
    return dest


def file_signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@contextmanager
def output_lock(path):
    with path.with_name(path.name+'.fill.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another fill process is writing {path}') from None
        yield


def atomic_features(path, features):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as f:
            torch.save(features, f)
            f.flush()
            # Commit a complete tensor file atomically.
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def feature_lock(directory):
    """One generator writes a shared feature directory at a time."""
    with (Path(directory) / '.generate.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another generator is writing {directory}') from None
        yield

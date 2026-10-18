"""
Operational error handling & text sanitization tools.
"""
from __future__ import annotations

class ScanBusyError(Exception):
    """Raised when another scan is already in progress."""
    pass

def safe_error_text(exc: Exception | str, max_len: int = 300, mask_tokens: tuple[str, ...] = ()) -> str:
    msg = str(exc)
    for token in mask_tokens:
        if token and isinstance(token, str):
            msg = msg.replace(token, "*****")
    return msg[:max_len]

from contextlib import contextmanager
from pathlib import Path
import os
import time


@contextmanager
def scan_lock(path: Path, stale_seconds: int = 3600):
    """Simple cross-rerun lock for one expensive market scan.

    Streamlit can rerun the script while a scan is active. An exclusive lock file
    prevents a second scan from writing the same SQLite/dashboard files. Stale
    locks are removed after ``stale_seconds``.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}".encode("ascii", errors="ignore"))
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                age = 0
            # A dead server process can leave an exclusive lock behind.  On
            # reboot, recover immediately rather than waiting a full hour;
            # never remove the lock of a living worker just because it is old.
            orphaned = False
            has_valid_pid = False
            try:
                content = lock_path.read_text(encoding="ascii", errors="ignore")
                pid_text = content.split("pid=", 1)[1].split(" ", 1)[0]
                pid = int(pid_text)
                if pid > 0:
                    has_valid_pid = True
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        orphaned = True
                    except PermissionError:
                        orphaned = False
            except (OSError, ValueError, IndexError):
                pass
            if attempt == 0 and (orphaned or (not has_valid_pid and age > max(60, int(stale_seconds)))):
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            raise ScanBusyError("Another market scan is already running")
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

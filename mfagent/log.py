"""Server-safe logging: never raises, and writes at most a few kilobytes in total.

The official harness aborts a whole task run when a request fails or hangs. A print
to a closed pipe raises BrokenPipeError, and a pipe that nobody reads blocks once
about 64 kB are buffered, so all messages go through ``log``: errors are swallowed
and the total volume stays far below a pipe buffer.
"""
from __future__ import annotations

import sys
import threading

MAX_BYTES = 16_000
MAX_LINE = 1_500

_lock = threading.Lock()
_written = 0


def log(msg: str) -> None:
    global _written
    try:
        with _lock:
            if _written >= MAX_BYTES:
                return
            line = f"[mfagent] {msg}"
            if len(line) > MAX_LINE:
                line = line[:MAX_LINE] + " ...[truncated]"
            if _written + len(line) + 1 > MAX_BYTES:
                line = "[mfagent] log volume limit reached; later messages are dropped"
                _written = MAX_BYTES
            else:
                _written += len(line) + 1
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
    except Exception:
        pass

"""
Tracks when engagement state was last requested (dashboard/state polling).
Used by the detection loop to reduce frame processing when no clients are viewing.
"""

import time
import threading

_last_request_time: float = 0.0
_lock = threading.Lock()
_IDLE_THRESHOLD_SEC = 60.0


def update_last_request() -> None:
    """Call when /engagement/state or similar is requested. Thread-safe."""
    global _last_request_time
    with _lock:
        _last_request_time = time.time()


def get_last_request_time() -> float:
    """Return timestamp of last request. Thread-safe."""
    with _lock:
        return _last_request_time


def is_idle(threshold_sec: float = _IDLE_THRESHOLD_SEC) -> bool:
    """True if no engagement request in the last threshold_sec seconds. False if never requested."""
    with _lock:
        if _last_request_time == 0.0:
            return False
        return (time.time() - _last_request_time) > threshold_sec

"""
Detection worker process scaffold.

When DETECTION_WORKER_PROCESS=true, engagement detection can run in a separate
process to isolate CPU-heavy work (MediaPipe, feature extraction) from the Flask
HTTP process, improving responsiveness and memory isolation.

Architecture (when implemented):
- Main process: VideoSourceHandler reads frames, sends (frame, timestamp) to input Queue
- Worker process: Receives frames, runs EngagementStateDetector._process_frame logic,
  sends (state_dict, alert) to output Queue. Initializes detector once on start.
- Main process: Receives state, updates engagement_detector.current_state, handles alerts.

Benefits: Flask stays responsive under load; worker can be restarted independently;
memory for ML models is isolated in worker process.

To implement: extend DetectionWorkerProcess.run_loop() to init detector, process
frames from self._input_queue, put results on self._output_queue. Wire
engagement_state_detector to use this when config.DETECTION_WORKER_PROCESS is True.
"""

import multiprocessing as mp
from typing import Any, Optional

import config


def should_use_worker_process() -> bool:
    """True if detection should run in a separate process."""
    return getattr(config, "DETECTION_WORKER_PROCESS", False)


class DetectionWorkerProcess:
    """
    Scaffold for running engagement detection in a subprocess.

    Usage (when implemented):
        worker = DetectionWorkerProcess()
        worker.start()
        worker.put_frame(frame_bgr)
        state = worker.get_state()
        worker.stop()
    """

    def __init__(self):
        self._process: Optional[mp.Process] = None
        self._input_queue: Optional[mp.Queue] = None
        self._output_queue: Optional[mp.Queue] = None

    def start(self, detection_method: str = "mediapipe", lightweight: bool = False) -> bool:
        """Start the worker process. Returns True if started."""
        if not should_use_worker_process():
            return False
        raise NotImplementedError(
            "DETECTION_WORKER_PROCESS is enabled but detection_worker is not fully implemented. "
            "Keep DETECTION_WORKER_PROCESS=false or implement run_loop() in this module."
        )

    def stop(self) -> None:
        """Stop the worker process."""
        pass

    def put_frame(self, frame: Any) -> None:
        """Send a frame to the worker for processing."""
        if self._input_queue is not None:
            self._input_queue.put(frame)

    def get_state(self, timeout: float = 0.01) -> Optional[dict]:
        """Get the latest state from the worker, or None if none available."""
        return None

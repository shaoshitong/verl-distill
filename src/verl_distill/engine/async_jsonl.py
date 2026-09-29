"""Bounded per-process JSONL sink; worker receives immutable CPU strings only."""
import json
import queue
import threading
import time
from pathlib import Path

class AsyncJSONL:
    def __init__(self, capacity=32, timeout=30.0):
        self.queue = queue.Queue(maxsize=capacity)
        self.timeout = timeout
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="scalar-jsonl", daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError("Asynchronous scalar log write failed") from self.error

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    path, text = item
                    with Path(path).open("a", encoding="utf-8") as stream:
                        stream.write(text)
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def enqueue(self, path, record):
        if self.closed:
            raise RuntimeError("Scalar logger already closed")
        self.check()
        # JSON rejects tensors/non-finite values before any background work.
        text = json.dumps(record, allow_nan=False) + "\n"
        deadline = time.monotonic() + self.timeout
        while True:
            self.check()
            try:
                self.queue.put((str(path), text), timeout=min(.1, max(0, deadline-time.monotonic())))
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Scalar logger queue remained full")
        self.check()

    def flush(self):
        deadline = time.monotonic() + self.timeout
        while self.queue.unfinished_tasks:
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("Scalar logger drain timed out")
            time.sleep(.01)
        self.check()

    def close(self):
        if self.closed:
            self.check()
            return
        try:
            self.flush()
        finally:
            self.closed = True
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass  # Stalled daemon cannot block process teardown.
        self.thread.join(timeout=self.timeout)
        if self.thread.is_alive():
            raise TimeoutError("Scalar logger worker did not exit")
        self.check()

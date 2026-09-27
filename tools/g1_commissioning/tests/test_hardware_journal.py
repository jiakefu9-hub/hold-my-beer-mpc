"""Offline recorder failure/drain tests. No SDK imports or robot endpoints."""
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_walk_pid import Journal


class JournalTest(unittest.TestCase):
    def test_enqueue_between_empty_get_and_close_is_drained(self):
        observed_empty = threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("g1_walk_pid.threading.Thread.start"):
                log = Journal(Path(folder) / "run")

            class RacingQueue(queue.Queue):
                first_get = True

                def get(self, block=True, timeout=None):
                    if self.first_get:
                        self.first_get = False
                        # Pause after observing an empty queue, before the
                        # writer's Empty handler. close() releases this wait.
                        observed_empty.set()
                        if not log._closing.wait(2.):
                            raise RuntimeError("test close was not called")
                        raise queue.Empty
                    return super().get(block=block, timeout=timeout)

            log._queue = RacingQueue()
            log._thread.start()
            try:
                self.assertTrue(observed_empty.wait(1.))
                log.record({"last_accepted_row": True})
            finally:
                closed = log.close(timeout_s=3.)
            self.assertTrue(closed)
            self.assertEqual(log.written, 1)
            self.assertTrue(log._queue.empty())
            rows = [json.loads(s) for s in (log.output_dir / "raw.jsonl").read_text().splitlines()]
            self.assertTrue(rows[0]["last_accepted_row"])

    def test_drain_and_idempotent_close(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Journal(Path(folder) / "run")
            for i in range(100):
                log.record({"sequence": i, "valid": np.bool_(True)})
            self.assertTrue(log.close())
            self.assertTrue(log.close())
            rows = [json.loads(line) for line in (log.output_dir / "raw.jsonl").read_text().splitlines()]
            self.assertEqual([r["sequence"] for r in rows], list(range(100)))
            self.assertTrue(all(r["valid"] is True for r in rows))
            self.assertFalse(log._thread.is_alive())
            self.assertTrue(log._stream.closed)
            log.record({"late": True})
            self.assertTrue(log.failed.is_set())
            self.assertEqual(log.dropped, 1)

    def test_full_queue_and_writer_error_never_blocks_shutdown(self):
        # Prevent the writer from starting until its tiny queue is full. A
        # sentinel-based blocking put could hang once the writer then fails.
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("g1_walk_pid.threading.Thread.start"):
                log = Journal(Path(folder) / "run")
            log._queue = queue.Queue(maxsize=1)
            log.record({"not_json_serializable": object()})
            log.record({"overflow": True})
            log._thread.start()
            self.assertFalse(log.close(timeout_s=1.))
            self.assertFalse(log._thread.is_alive())
            self.assertTrue(log.failed.is_set())
            self.assertEqual(log.dropped, 1)
            self.assertTrue(log._stream.closed)

    def test_blocked_writer_is_marked_incomplete_without_cross_thread_close(self):
        entered, unblock = threading.Event(), threading.Event()

        class SlowStream:
            closed = False

            def write(self, text):
                entered.set()
                unblock.wait(2.)

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch("g1_walk_pid.threading.Thread.start"):
                log = Journal(Path(folder) / "run")
            log._stream.close()
            stream = log._stream = SlowStream()
            log._thread.start()
            try:
                log.record({"slow": True})
                self.assertTrue(entered.wait(1.))
                start = time.monotonic()
                self.assertFalse(log.close(timeout_s=.03))
                self.assertLess(time.monotonic()-start, .5)
                self.assertIn("timed out", log.failure_reason)
                self.assertFalse(stream.closed)
            finally:
                unblock.set()
                log._thread.join(1.)
            self.assertTrue(stream.closed)
            self.assertFalse(log.close())  # timeout cannot become success later


if __name__ == "__main__":
    unittest.main()

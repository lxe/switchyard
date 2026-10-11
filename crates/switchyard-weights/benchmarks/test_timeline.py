"""Clock alignment and concurrent file reception checks for the live timeline."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from timeline import Timeline


class TimelineTests(unittest.TestCase):
    def test_native_clock_and_sanitization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            timeline = Timeline(path, 100.0)
            timeline.native(
                {
                    "kind": "operation",
                    "name": "cuda_init",
                    "pid": 3,
                    "started_monotonic": 101.0,
                    "ended_monotonic": 102.5,
                    "caller_stack": [{"file": "private-path"}],
                    "secret": "private-value",
                }
            )
            timeline.close()
            item = json.loads(path.read_text())
            self.assertEqual(item["elapsed_s"], 2.5)
            self.assertEqual(item["duration_s"], 1.5)
            self.assertNotIn("private", path.read_text())

    def test_polled_milestones_are_emitted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            timeline = Timeline(path, 100.0)
            state = {
                "startup_milestones": [
                    {"event": "runtime_boot_ready", "elapsed_s": 0.25}
                ]
            }
            timeline.state(state, 102.0)
            timeline.state(state, 102.0)
            timeline.close()
            items = path.read_text().splitlines()
            self.assertEqual(len(items), 1)
            self.assertEqual(json.loads(items[0])["elapsed_s"], 2.25)

    def test_partial_line_and_final_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native.jsonl"
            timeline = Timeline(root / "events.jsonl", 100.0)
            stop = threading.Event()
            thread = threading.Thread(
                target=timeline.follow_native_file, args=(source, stop)
            )
            thread.start()
            try:
                source.write_text('{"kind":"process_start",')
                time.sleep(0.15)
                with source.open("a") as stream:
                    stream.write('"monotonic":101.5}\n')
            finally:
                stop.set()
                thread.join(2)
                timeline.close()
            self.assertFalse(thread.is_alive())
            items = (root / "events.jsonl").read_text().splitlines()
            self.assertEqual(len(items), 1)
            self.assertEqual(json.loads(items[0])["elapsed_s"], 1.5)


if __name__ == "__main__":
    unittest.main()

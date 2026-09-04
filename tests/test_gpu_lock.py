import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_LOCK = REPO_ROOT / "gpu-lock"
TIMEOUT_SECONDS = 10.0

FIRST_COMMAND = """
from pathlib import Path
import sys
import time

events = Path(sys.argv[1])
release = Path(sys.argv[2])
with events.open("a", encoding="utf-8") as stream:
    stream.write("first-start\\n")
    stream.flush()
while not release.exists():
    time.sleep(0.01)
with events.open("a", encoding="utf-8") as stream:
    stream.write("first-end\\n")
"""

SECOND_COMMAND = """
from pathlib import Path
import sys

with Path(sys.argv[1]).open("a", encoding="utf-8") as stream:
    stream.write("second-start\\n")
"""


def wait_for_file_line(path: Path, expected: str) -> None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.exists() and expected in path.read_text(encoding="utf-8").splitlines():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {expected!r} in {path}")


def wait_for_stderr_line(process: subprocess.Popen[str], expected: str) -> str:
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            events = selector.select(deadline - time.monotonic())
            if not events:
                break
            line = process.stderr.readline()
            if expected in line:
                return line
            if not line and process.poll() is not None:
                break
    finally:
        selector.close()
    raise AssertionError(
        f"timed out waiting for {expected!r}; returncode={process.poll()}"
    )


class GpuLockTests(unittest.TestCase):
    def test_two_wrappers_serialize_on_one_persistent_inode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qw gpu lock ") as temporary:
            home = Path(temporary)
            events = home / "events with spaces.txt"
            release = home / "release first holder"
            holder_env = os.environ | {
                "HOME": os.fspath(home),
                "QW_GPU_LOCK_SESSION": "holder session",
            }
            waiter_env = os.environ | {
                "HOME": os.fspath(home),
                "QW_GPU_LOCK_SESSION": "waiter session",
            }
            first = subprocess.Popen(
                [
                    os.fspath(GPU_LOCK),
                    "--",
                    sys.executable,
                    "-c",
                    FIRST_COMMAND,
                    os.fspath(events),
                    os.fspath(release),
                ],
                env=holder_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            second = None
            try:
                wait_for_file_line(events, "first-start")
                second = subprocess.Popen(
                    [
                        os.fspath(GPU_LOCK),
                        "--",
                        sys.executable,
                        "-c",
                        SECOND_COMMAND,
                        os.fspath(events),
                    ],
                    env=waiter_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                waiting = wait_for_stderr_line(second, "gpu-lock: waiting")
                self.assertIn('"pid":', waiting)
                self.assertIn('"session": "QW_GPU_LOCK_SESSION=holder session"', waiting)
                self.assertEqual(events.read_text(encoding="utf-8"), "first-start\n")

                release.touch()
                first_stderr = first.communicate(timeout=TIMEOUT_SECONDS)[1]
                second_stderr = second.communicate(timeout=TIMEOUT_SECONDS)[1]
                self.assertEqual(first.returncode, 0, first_stderr)
                self.assertEqual(second.returncode, 0, second_stderr)
                self.assertEqual(
                    events.read_text(encoding="utf-8").splitlines(),
                    ["first-start", "first-end", "second-start"],
                )
                self.assertTrue((home / ".cache" / "qw" / "gpu_lock").is_file())
            finally:
                release.touch(exist_ok=True)
                for process in (second, first):
                    if process is not None and process.poll() is None:
                        process.terminate()
                        process.communicate(timeout=TIMEOUT_SECONDS)

    def test_child_exit_status_is_the_wrapper_exit_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qw gpu lock status ") as temporary:
            home = Path(temporary)
            completed = subprocess.run(
                [
                    os.fspath(GPU_LOCK),
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(37)",
                ],
                env=os.environ | {"HOME": os.fspath(home)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 37, completed.stderr)
            self.assertIn("gpu-lock: acquired", completed.stderr)
            self.assertTrue((home / ".cache" / "qw" / "gpu_lock").is_file())


if __name__ == "__main__":
    unittest.main()

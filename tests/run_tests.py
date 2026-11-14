#!/usr/bin/env python3
"""
Runs the sequence files next to this script through the sample in headless mode.

Every sequence file walks a set of configurations, the sample re-initializes the renderer
whenever one of them changes. A test fails when the sample reports a validation error, a
shader compilation error, a failed renderer init, a lost device, when it produced fewer
screenshots than it has sequences, or when it doesn't exit cleanly. See tests/README.md.

  python tests/run_tests.py
  python tests/run_tests.py --filter hbao --verbose
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# these show up in the sample's log output and mean the run is broken
FAILURE_PATTERNS = [
    re.compile(r"Validation Error", re.IGNORECASE),
    re.compile(r"\berror:", re.IGNORECASE),  # shader compilation
    re.compile(r"VUID-"),
    re.compile(r"shaders? failed", re.IGNORECASE),
    re.compile(r"Renderer init failed", re.IGNORECASE),
    re.compile(r"DEVICE_LOST", re.IGNORECASE),
    re.compile(r"failed to allocate", re.IGNORECASE),
    re.compile(r"Resources exceeding max buf", re.IGNORECASE),
]

def find_exe(rootDir, explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"executable not found: {path}")
        return path

    # named after the checkout's folder, as in CMakeLists.txt
    def existing(configs):
        paths = [rootDir / "_bin" / config / name for config in configs for name in (rootDir.name + ".exe", rootDir.name)]
        return [path for path in paths if path.is_file()]

    # the newer of the optimized builds, Debug only without them
    candidates = existing(["Release", "RelWithDebInfo"]) or existing(["Debug"])
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    sys.exit(f"no executable found below {rootDir}, provide one via --exe")


def count_sequences(sequenceFile):
    count = 0
    for line in sequenceFile.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("SEQUENCE"):
            count += 1
    return count


def run_sequence_file(exe, sequenceFile, args, resultDir):
    # the sequencer closes the application after the last sequence, the frame budget just
    # has to be generous enough to get there
    headlessFrames = count_sequences(sequenceFile) * args.sequenceframes + 64

    cmdline = [
        str(exe),
        "--headless",
        "--headlessframes", str(headlessFrames),
        "--windowsize", str(args.width), str(args.height),
        "--supersample", "1",
        "--sequencefile", str(sequenceFile),
    ]
    if args.screenshots:
        # rendered viewport at the end of each sequence, named after its index and description
        cmdline += ["--sequencescreenshot", "2"]
    if args.validation:
        cmdline += ["--validation", "1"]
    if args.scene:
        cmdline += ["--scene", args.scene]
    cmdline += args.extra

    start = time.monotonic()
    try:
        completed = subprocess.run(cmdline, capture_output=True, text=True, timeout=args.timeout, cwd=resultDir)
        output = completed.stdout + completed.stderr
        returnCode = completed.returncode
    except subprocess.TimeoutExpired as timeout:
        # the streams are bytes on a timeout, even with text=True
        def as_text(stream):
            if isinstance(stream, bytes):
                return stream.decode(errors="replace")
            return stream or ""

        output = as_text(timeout.stdout) + as_text(timeout.stderr)
        output += f"\ntimeout after {args.timeout} seconds\n"
        returnCode = -1

    duration = time.monotonic() - start

    (resultDir / "output.log").write_text(output, encoding="utf-8", errors="replace")

    failures = []
    if args.screenshots:
        numShots = len(list(resultDir.glob("screenshot_*.jpg")))
        numSequences = count_sequences(sequenceFile)
        if numShots < numSequences:
            failures.append(f"{numShots} screenshots for {numSequences} sequences")
    if returnCode != 0:
        failures.append(f"exit code {returnCode}")
    for pattern in FAILURE_PATTERNS:
        match = pattern.search(output)
        if match:
            line = next((l.strip() for l in output.splitlines() if pattern.search(l)), match.group(0))
            failures.append(line[:160])

    return failures, duration, output, cmdline


def main():
    rootDir = Path(__file__).resolve().parent.parent
    testsDir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="run the sequence file based tests")
    parser.add_argument("--exe", help="path to the sample executable")
    parser.add_argument("--filter", default="", help="only run sequence files whose name contains this")
    parser.add_argument("--scene", default="", help="scene file, defaults to the sample's own default")
    parser.add_argument("--width", type=int, default=1024, help="headless viewport width, default 1024")
    parser.add_argument("--height", type=int, default=768, help="headless viewport height, default 768")
    parser.add_argument("--sequenceframes", type=int, default=8,
                        help="frames per sequence used to size the headless frame budget, default 8")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per sequence file, default 600")
    parser.add_argument("--no-screenshots", dest="screenshots", action="store_false",
                        help="skip the screenshot at the end of each sequence")
    parser.add_argument("--no-validation", dest="validation", action="store_false",
                        help="skip the Vulkan validation layers")
    parser.add_argument("--verbose", action="store_true", help="print the sample's output")
    parser.add_argument("extra", nargs="*", help="extra arguments passed to the sample")
    args = parser.parse_args()

    exe = find_exe(rootDir, args.exe)
    sequenceFiles = sorted(f for f in testsDir.glob("*.txt") if args.filter in f.name)
    if not sequenceFiles:
        sys.exit(f"no sequence files matching '{args.filter}' in {testsDir}")

    print(f"executable: {exe} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exe.stat().st_mtime))})")
    print(f"sequences:  {len(sequenceFiles)} file(s), validation {'on' if args.validation else 'off'}\n")

    # screenshots and logs end up in tests/_results/<timestamp>/<sequence file>/
    resultsDir = testsDir / "_results" / time.strftime("%Y%m%d_%H%M%S")
    print(f"results:    {resultsDir}")

    numFailed = 0
    for sequenceFile in sequenceFiles:
        resultDir = resultsDir / sequenceFile.stem
        resultDir.mkdir(parents=True, exist_ok=True)
        print(f"  {sequenceFile.name:<28}", end="", flush=True)
        failures, duration, output, cmdline = run_sequence_file(exe, sequenceFile, args, resultDir)

        if failures:
            numFailed += 1
            print(f"FAILED  ({duration:.1f}s)")
            for failure in failures:
                print(f"      {failure}")
            print(f"      {' '.join(cmdline)}")
        else:
            print(f"ok      ({duration:.1f}s, {count_sequences(sequenceFile)} sequences)")

        if args.verbose:
            print(output)

    print()
    if numFailed:
        print(f"{numFailed} of {len(sequenceFiles)} sequence files FAILED")
        return 1

    print(f"all {len(sequenceFiles)} sequence files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

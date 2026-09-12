#!/usr/bin/env python3
"""Fetch the exact non-redistributable source photos into git-ignored local/."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-source-rights", action="store_true",
        help="Acknowledge source copyright restrictions; downloading grants no redistribution rights.",
    )
    args = parser.parse_args()
    if not args.accept_source_rights:
        parser.error("Review manifest.json rights_status and pass --accept-source-rights for local use.")
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    for fixture in manifest["fixtures"]:
        if fixture["kind"] != "source_photo":
            continue
        destination = directory / fixture["path"]
        expected = fixture["sha256"]
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise SystemExit(f"Existing file differs; preserving {destination}")
        else:
            request = urllib.request.Request(fixture["source_url"], headers={"User-Agent": "qwr-fixture-fetch/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise SystemExit(f"Source changed for {fixture['id']}; not writing unverified bytes")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        print(destination)


if __name__ == "__main__":
    main()

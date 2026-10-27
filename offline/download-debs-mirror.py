#!/usr/bin/env python3
"""Download Ubuntu 24.04 (noble) amd64 .debs without apt-get (for offline bundles)."""
from __future__ import annotations

import argparse
import gzip
import re
import ssl
import urllib.request
from collections import deque
from pathlib import Path

ARCHIVE = "http://archive.ubuntu.com/ubuntu"
SUITES = (
    ("noble", "main"),
    ("noble", "universe"),
    ("noble-updates", "main"),
    ("noble-updates", "universe"),
)

SEED_PKGS = (
    "python3.12",
    "python3.12-venv",
    "python3.12-dev",
    "python3-pip",
    "build-essential",
    "ninja-build",
    "git",
    "libgl1",
    "libglib2.0-0",
    "libsm6",
    "libxext6",
    "libxrender1",
    "libgomp1",
)


def load_packages() -> dict[str, dict]:
    index: dict[str, dict] = {}
    ctx = ssl.create_default_context()
    for suite, component in SUITES:
        url = f"{ARCHIVE}/dists/{suite}/{component}/binary-amd64/Packages.gz"
        print(f"==> Index {suite}/{component}")
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
            raw = gzip.decompress(resp.read()).decode("utf-8", errors="replace")
        for stanza in raw.split("\n\n"):
            fields: dict[str, str] = {}
            for line in stanza.splitlines():
                if ": " not in line or line.startswith(" "):
                    continue
                key, val = line.split(": ", 1)
                fields[key] = val
            name = fields.get("Package")
            filename = fields.get("Filename")
            if not name or not filename:
                continue
            prio = 2 if "updates" in suite else 1
            prev = index.get(name)
            if prev is None or prio >= prev.get("_prio", 0):
                index[name] = {
                    "Filename": filename,
                    "Depends": fields.get("Depends", ""),
                    "_prio": prio,
                }
    return index


def parse_depends(depends: str) -> list[str]:
    if not depends:
        return []
    names: list[str] = []
    for part in depends.split(","):
        part = part.strip()
        if not part:
            continue
        if "|" in part:
            part = part.split("|", 1)[0].strip()
        name = re.split(r"\s*[<(]", part, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def collect_deps(index: dict[str, dict], seeds: tuple[str, ...]) -> set[str]:
    needed: set[str] = set()
    queue: deque[str] = deque(seeds)
    while queue:
        pkg = queue.popleft()
        if pkg in needed:
            continue
        if pkg not in index:
            print(f"  (missing in index: {pkg})")
            continue
        needed.add(pkg)
        for dep in parse_depends(index[pkg].get("Depends", "")):
            if dep not in needed:
                queue.append(dep)
    return needed


def download_all(out: Path, packages: set[str], index: dict[str, dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    ordered = sorted(packages)
    for i, pkg in enumerate(ordered, 1):
        meta = index.get(pkg)
        if not meta:
            print(f"  (skip: {pkg})")
            continue
        filename = meta["Filename"]
        url = f"{ARCHIVE}/{filename}"
        dest = out / Path(filename).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[{i}/{len(ordered)}] {pkg}")
        req = urllib.request.Request(url, headers={"User-Agent": "ksb-ai-offline/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=300) as resp:
            dest.write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Destination directory for .deb files")
    args = parser.parse_args()
    index = load_packages()
    packages = collect_deps(index, SEED_PKGS)
    print(f"==> Downloading {len(packages)} packages → {args.out}")
    download_all(args.out, packages, index)
    print("Done.")


if __name__ == "__main__":
    main()

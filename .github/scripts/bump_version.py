#!/usr/bin/env python3
"""Bump the `version = "X.Y.Z"` field in a pyproject.toml (PEP 440).

Usage:
    python bump_version.py <major|minor|patch> path/to/pyproject.toml

Semantics (current -> new):
    major : X.Y.Z -> (X+1).0.0
    minor : X.Y.Z -> X.(Y+1).0
    patch : X.Y.Z -> X.Y.(Z+1)

Prints the new version string to stdout.
"""
import re
import sys

VERSION_RE = re.compile(r'^(?P<pre>version\s*=\s*")(?P<ver>[^"]+)(?P<post>")', re.M)
PARSE_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def bump(current: str, kind: str) -> str:
    m = PARSE_RE.match(current)
    if not m:
        sys.exit(f"Cannot parse version {current!r} (expected X.Y.Z)")
    major, minor, patch = int(m[1]), int(m[2]), int(m[3])

    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    sys.exit(f"Unknown bump kind {kind!r}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    kind, path = sys.argv[1], sys.argv[2]

    text = path_text = open(path, encoding="utf-8").read()
    match = VERSION_RE.search(text)
    if not match:
        sys.exit(f"No `version = \"...\"` field found in {path}")

    new_version = bump(match["ver"], kind)
    text = VERSION_RE.sub(lambda m: f'{m["pre"]}{new_version}{m["post"]}', path_text, count=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(new_version)


if __name__ == "__main__":
    main()

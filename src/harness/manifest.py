"""Run manifests (ARCHITECTURE.md: "every run writes a manifest"). One JSON file per run_id, written
once — like TREC run files (trec.py), manifests are append-only: a script that's re-run against
the same config should fail loudly, not quietly overwrite the record of what happened last
time. Timestamps are injected by the caller (not read here) so this module stays trivially
testable without mocking the clock.
"""
from __future__ import annotations

import json
import os


def write_manifest(path: str, manifest: dict) -> None:
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} exists — manifests are append-only, same as run artifacts; "
            "new config -> new run_id -> new manifest path"
        )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def read_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

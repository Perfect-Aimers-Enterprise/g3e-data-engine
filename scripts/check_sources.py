#!/usr/bin/env python3
"""
Run the preflight check WITHOUT running the pipeline — exactly the check
Pipeline.run(dry_run=False) does before downloading anything, exposed as
its own fast (no network), standalone command. Run this right after editing
configs/datasets.yaml, or right after setting up a new environment (Colab,
a fresh cloud box, etc.) — it catches a missing downloader dependency
(e.g. `roboflow` not installed) in milliseconds instead of after minutes or
hours of downloading other sources first.

Exit code 0 = every enabled source is ready. Exit code 1 = at least one
isn't (missing dependency, missing repo/project reference, or an
unverified license).
"""
from __future__ import annotations

from g3e_data_engine import load_engine_config
from g3e_data_engine.core.preflight import run_preflight


def main() -> int:
    cfg = load_engine_config()
    report = run_preflight(cfg)
    print(report.render())
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

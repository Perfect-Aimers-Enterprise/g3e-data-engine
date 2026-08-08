#!/usr/bin/env python3
"""
Check every enabled source's readiness WITHOUT running the pipeline —
i.e. exactly the check Pipeline.run(dry_run=False) does before downloading
anything, exposed as its own command so you can validate configs/datasets.yaml
after editing it.

Exit code 0 = all enabled sources are ready. Exit code 1 = at least one isn't
(missing repo/project reference, or an unverified license).
"""
from __future__ import annotations

from g3e_data_engine import load_engine_config
from g3e_data_engine.core.exceptions import SourceConfigError


def main() -> int:
    cfg = load_engine_config()
    try:
        cfg.validate_sources_ready()
    except SourceConfigError as exc:
        print(exc)
        return 1

    print("G3E DATA ENGINE\n")
    for name in cfg.datasets.enabled_sources():
        print(f"\u2713 {name} — ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

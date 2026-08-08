"""
Preflight checks — verify every enabled source is actually able to run
BEFORE any network call for real data begins.

This exists because of a real failure mode: a run got most of the way
through resolving Hugging Face sources, then crashed on `from roboflow
import Roboflow` because the `roboflow` package simply wasn't installed in
that environment — an hour in, for something a single `importlib` check
would have caught in milliseconds. Preflight checks three things per
enabled source, and only three:

    1. dependency  — is the package `kind` needs actually importable?
    2. repository  — does it have a repo/project reference set?
    3. license     — has the license been verified?

Any failure on any source aborts the WHOLE run before a single byte is
downloaded, with a report naming exactly what's wrong and how to fix it.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from g3e_data_engine.core.exceptions import SourceConfigError

# kind -> (import name, install hint). Extend this one line when you add a
# new downloader kind — see docs/ARCHITECTURE.md "Adding a brand-new kind
# of source". A kind with no entry here is assumed dependency-free (no
# check performed) rather than failing closed, since most future kinds
# (e.g. a local-folder importer) may need nothing extra installed.
KIND_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "huggingface": ("datasets", "pip install datasets huggingface_hub"),
    "roboflow": ("roboflow", 'pip install "g3e-data-engine[roboflow]"'),
}


@dataclass
class SourceCheck:
    name: str
    kind: str
    ok: bool
    dependency_ok: bool
    repository_ok: bool
    license_ok: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    checks: list[SourceCheck]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = ["G3E DATA ENGINE \u2014 PREFLIGHT", "", "Sources", "\u2500" * 28]
        for c in self.checks:
            mark = "\u2713" if c.ok else "\u2717"
            lines.append(f"{mark} {c.name}")

            dep_mark = "\u2713" if c.dependency_ok else "\u2717"
            lines.append(f"  {c.kind + ' dependency':<26} {dep_mark}")

            if c.dependency_ok:
                repo_mark = "\u2713" if c.repository_ok else "\u2717"
                lines.append(f"  {'repository':<26} {repo_mark}")
                lic_mark = "\u2713" if c.license_ok else "\u2717"
                lines.append(f"  {'license':<26} {lic_mark}")
            lines.append("")

        failing = [c for c in self.checks if not c.ok]
        if failing:
            lines.append("ERROR")
            for c in failing:
                for err in c.errors:
                    lines.append(err)
                    lines.append("")
            lines.append("Processing aborted. No data was downloaded.")
        else:
            lines.append("All enabled sources are ready.")

        return "\n".join(lines)


def run_preflight(config, source_names: list[str] | None = None) -> PreflightReport:
    """
    Pure/offline — makes no network calls, only `importlib.import_module`
    (which is effectively free if the package is installed, and fails
    instantly if it isn't) and config field checks.
    """
    sources = config.datasets.enabled_sources()
    if source_names is not None:
        sources = {k: v for k, v in sources.items() if k in source_names}

    checks: list[SourceCheck] = []
    for name, src in sources.items():
        errors: list[str] = []

        dependency_ok = True
        dep_import, dep_hint = KIND_DEPENDENCIES.get(src.kind, (None, None))
        if dep_import:
            try:
                importlib.import_module(dep_import)
            except ImportError:
                dependency_ok = False
                errors.append(
                    f"\u2717 {name}\n"
                    f"  {src.kind.capitalize()} source is enabled but the `{dep_import}` "
                    f"package is missing.\n\n  Install:\n  {dep_hint}"
                )

        repository_ok = True
        if dependency_ok:
            if src.kind == "huggingface" and not src.hf_repo:
                repository_ok = False
                errors.append(f"\u2717 {name}\n  Source enabled but hf_repo is missing.")
            elif src.kind == "roboflow" and (not src.project or src.version is None):
                repository_ok = False
                errors.append(
                    f"\u2717 {name}\n  Source enabled but project/version is missing "
                    "(pin both — don't float a version)."
                )

        license_ok = src.license.verified
        if dependency_ok and not license_ok:
            errors.append(
                f"\u2717 {name}\n  Dataset license has not been verified "
                f"(name={src.license.name!r}). Set license.verified: true in "
                "configs/datasets.yaml after reviewing the actual terms."
            )

        ok = dependency_ok and repository_ok and license_ok
        checks.append(
            SourceCheck(
                name=name,
                kind=src.kind,
                ok=ok,
                dependency_ok=dependency_ok,
                repository_ok=repository_ok,
                license_ok=license_ok,
                errors=errors,
            )
        )

    return PreflightReport(checks=checks)


def run_preflight_or_raise(config, source_names: list[str] | None = None) -> None:
    report = run_preflight(config, source_names)
    if not report.all_ok:
        raise SourceConfigError(report.render())

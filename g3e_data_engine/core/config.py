"""
Configuration models and loader for g3e-data-engine.

Every stage of the pipeline reads its settings from here rather than from
hardcoded constants. See configs/*.yaml for the actual values, and
DATASET_SPEC.md for what each field means and why it exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from g3e_data_engine.core.exceptions import SourceConfigError

# Repo root: g3e_data_engine/core/config.py -> core -> g3e_data_engine -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# classes.yaml
# ---------------------------------------------------------------------------
class ClassDef(BaseModel):
    name: str
    id: int
    priority_tier: int = Field(ge=1, le=4)
    note: str = ""


class ClassesConfig(BaseModel):
    version: str
    classes: list[ClassDef]

    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    def by_name(self, name: str) -> ClassDef:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"Unknown class: {name!r}. Known: {self.names()}")


# ---------------------------------------------------------------------------
# datasets.yaml
# ---------------------------------------------------------------------------
class LicenseInfo(BaseModel):
    """
    Machine-readable license status for a source.

    `verified` is deliberately NOT inferred from anything (not from the
    source's own claims, not from a similar dataset's license) — a human
    reviews the actual terms and flips this to True. Until then, an enabled
    source with `verified: False` is refused before any download happens.
    See DATASET_SPEC.md section 11.
    """
    name: str = "UNKNOWN"
    verified: bool = False
    url: str = ""


class AuthConfig(BaseModel):
    """
    Credential lookup config for a source. The actual token/API key is NEVER
    stored here or in any yaml file — only *where to find it* is. See
    g3e_data_engine/core/credentials.py.
    """
    token_env: str | None = None  # overrides the kind's default env var name
    required: bool = False        # if True, downloading fails fast without a token


class SourceDef(BaseModel):
    enabled: bool = True
    kind: str = "huggingface"          # registry key — see downloader/base.py
    hf_repo: str = ""                  # used by kind="huggingface"
    project: str = ""                  # used by kind="roboflow", e.g. "workspace/project-slug"
    version: int | None = None         # used by kind="roboflow" — pin it, don't float
    classes: list[str] = Field(default_factory=list)
    class_map: dict[str, str] = Field(default_factory=dict)  # raw source label -> g3e class name
    max_images: int = 1000
    max_rows_scanned: int = 50_000  # safety valve — see downloader/hf_downloader.py DEFAULT_MAX_ROWS_SCANNED
    license: LicenseInfo = Field(default_factory=LicenseInfo)
    auth: AuthConfig = Field(default_factory=AuthConfig)


class DatasetsConfig(BaseModel):
    sources: dict[str, SourceDef]
    global_max_images: int = 8000

    def enabled_sources(self) -> dict[str, SourceDef]:
        return {k: v for k, v in self.sources.items() if v.enabled}


# ---------------------------------------------------------------------------
# processing.yaml
# ---------------------------------------------------------------------------
class ImageThresholds(BaseModel):
    min_shorter_side: int = 416
    blur_threshold: float = 90.0
    max_brightness: float = 250.0
    min_brightness: float = 35.0


class DuplicatesConfig(BaseModel):
    enabled: bool = True
    method: str = "phash"
    hamming_distance_threshold: int = 5


class LabelsConfig(BaseModel):
    remove_empty: bool = False


class SplitConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    @field_validator("test")
    @classmethod
    def _ratios_sum_to_one(cls, v, info):
        train = info.data.get("train", 0.8)
        val = info.data.get("val", 0.1)
        total = train + val + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {total}")
        return v


class ProcessingConfig(BaseModel):
    image: ImageThresholds
    duplicates: DuplicatesConfig
    labels: LabelsConfig
    split: SplitConfig


# ---------------------------------------------------------------------------
# priority.yaml
# ---------------------------------------------------------------------------
class BudgetConfig(BaseModel):
    total_images: int = 6000
    max_per_class: int = 2500
    min_per_class: int = 150


class PriorityConfig(BaseModel):
    budget: BudgetConfig
    tier_weights: dict[int, float]
    overrides: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregate engine config
# ---------------------------------------------------------------------------
class EngineConfig(BaseModel):
    classes: ClassesConfig
    datasets: DatasetsConfig
    processing: ProcessingConfig
    priority: PriorityConfig

    def validate_cross_refs(self) -> None:
        """Sanity checks that span multiple config files."""
        known = set(self.classes.names())
        for src_name, src in self.datasets.sources.items():
            unknown = set(src.classes) - known
            if unknown:
                raise ValueError(
                    f"datasets.yaml source '{src_name}' references unknown "
                    f"classes {unknown}. Known classes: {sorted(known)}"
                )
        if self.priority.budget.total_images > self.datasets.global_max_images:
            raise ValueError(
                "priority.yaml budget.total_images "
                f"({self.priority.budget.total_images}) exceeds "
                f"datasets.yaml global_max_images ({self.datasets.global_max_images}). "
                "Raise global_max_images deliberately if you really want a bigger run."
            )

    def validate_sources_ready(self, source_names: list[str] | None = None) -> None:
        """
        The "don't download anything until we're sure it's safe to" gate —
        now a preflight covering THREE things per enabled source, not just
        two: is the downloader dependency for its `kind` actually installed
        (this is what silently burns an hour discovering `roboflow` isn't in
        the environment), does it have a repo/project reference, and is its
        license verified.

        Unlike `validate_cross_refs` (checked on every config load, including
        dry runs and `/pipeline/allocate`), this is only called right before
        the download stage actually executes — so planning/allocation always
        stays available even while a source is still mid-setup. It runs in
        milliseconds (no network calls) so a bad environment or config is
        caught immediately, not after minutes/hours of downloading.

        Raises SourceConfigError listing every problem across every source
        at once, so a run never dies partway through, having already
        downloaded from source A only to fail on source B.
        """
        from g3e_data_engine.core.preflight import run_preflight_or_raise

        run_preflight_or_raise(self, source_names)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CONFIG_FILENAMES = ("classes.yaml", "datasets.yaml", "processing.yaml", "priority.yaml")
_config_cache: dict[str, tuple[tuple, EngineConfig]] = {}


def _fingerprint(base: Path) -> tuple:
    """
    (mtime_ns, size) per config file — NOT just a directory-path cache key.
    A plain `functools.lru_cache` keyed only on `configs_dir` was the exact
    bug behind a real failure: someone edited configs/datasets.yaml (e.g.
    flipping license.verified: true after a review) in the same running
    process/notebook that had already called load_engine_config() once —
    every subsequent call kept returning the pre-edit config, silently,
    because the cache had no way to know the files on disk had changed.
    This fingerprints the actual files so an edit is picked up automatically
    on the very next call, with no manual cache-clearing required.
    """
    parts = []
    for name in _CONFIG_FILENAMES:
        path = base / name
        if path.exists():
            st = path.stat()
            parts.append((name, st.st_mtime_ns, st.st_size))
        else:
            parts.append((name, None, None))
    return tuple(parts)


def load_engine_config(configs_dir: Optional[str] = None) -> EngineConfig:
    """
    Load and validate the full engine configuration from configs/*.yaml.

    Cached per `configs_dir`, but the cache is invalidated automatically
    whenever any of the four config files' modification time or size
    changes — so editing a yaml file and calling this again always sees the
    edit, even within the same long-running process (a notebook kernel, an
    API server, etc.). Repeated calls with unchanged files stay cheap
    (no re-parsing), so this costs nothing in the common case.
    """
    base = Path(configs_dir) if configs_dir else CONFIGS_DIR
    key = str(base)
    fingerprint = _fingerprint(base)

    cached = _config_cache.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    classes = ClassesConfig(**_read_yaml(base / "classes.yaml"))
    datasets = DatasetsConfig(**_read_yaml(base / "datasets.yaml"))
    processing = ProcessingConfig(**_read_yaml(base / "processing.yaml"))
    priority = PriorityConfig(**_read_yaml(base / "priority.yaml"))

    cfg = EngineConfig(
        classes=classes, datasets=datasets, processing=processing, priority=priority
    )
    cfg.validate_cross_refs()

    _config_cache[key] = (fingerprint, cfg)
    return cfg


def clear_config_cache() -> None:
    """
    Explicit escape hatch — mainly for tests that write config files and
    need a guaranteed-fresh read regardless of filesystem mtime resolution.
    Normal usage never needs to call this; editing a config file and calling
    load_engine_config() again is enough (see `_fingerprint`).
    """
    _config_cache.clear()

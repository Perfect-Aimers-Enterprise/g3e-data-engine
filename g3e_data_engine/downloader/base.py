"""
Downloader interface + registry.

Every source in configs/datasets.yaml has a `kind` field (e.g.
"huggingface"). That string is the registry key. To add a new source type
(e.g. a raw HTTP/zip source, a local-folder import, Roboflow, etc.), write a
class implementing `Downloader` and register it with `@register("your_kind")`
in its own module under g3e_data_engine/downloader/, then import that
module once from g3e_data_engine/downloader/__init__.py so it self-registers.

This file is one you should NOT need to touch when adding a new source —
see docs/ARCHITECTURE.md "Files to touch vs not touch".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DownloadedImage:
    """A single downloaded image + whatever boxes came with it, pre-conversion."""
    local_path: str
    source_name: str
    classes_present: list[str] = field(default_factory=list)
    raw_annotations: list[dict] = field(default_factory=list)  # source-native format


@dataclass
class DownloadRequest:
    source_name: str
    target_classes: dict[str, int]  # class_name -> how many images of this class to fetch
    dest_dir: str


class Downloader(ABC):
    """Base class every source-specific downloader implements."""

    @abstractmethod
    def download(self, request: DownloadRequest) -> list[DownloadedImage]:
        """
        Fetch up to `request.target_classes[c]` images per class `c` and
        return them as DownloadedImage records. Implementations MUST respect
        the per-class caps — this is the enforcement point for the priority
        budget computed by PriorityAllocator.
        """
        raise NotImplementedError


_REGISTRY: dict[str, type[Downloader]] = {}


def register(kind: str):
    def _decorator(cls: type[Downloader]):
        _REGISTRY[kind] = cls
        return cls

    return _decorator


def get_downloader(kind: str) -> Downloader:
    if kind not in _REGISTRY:
        raise KeyError(
            f"No downloader registered for kind={kind!r}. "
            f"Known kinds: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind]()

"""
Downloader registry package.

Importing this package registers every built-in downloader kind. Add new
`import` lines here when you add a new downloader module — that's the one
line you touch in this file for a new source type.
"""
from g3e_data_engine.downloader.base import Downloader, DownloadRequest, DownloadedImage, get_downloader

# Self-registering implementations:
from g3e_data_engine.downloader import hf_downloader  # noqa: F401
from g3e_data_engine.downloader import roboflow_downloader  # noqa: F401

__all__ = ["Downloader", "DownloadRequest", "DownloadedImage", "get_downloader"]

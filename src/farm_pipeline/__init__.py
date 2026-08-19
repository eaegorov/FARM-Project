"""Lightweight, model-independent tooling for reproducible FARM runs.

This package intentionally does not import :mod:`torch` or the online FARM
mapping stack.  Configuration and input validation must be usable before a
GPU container is started.
"""

from .scene_config import SceneConfig, SceneConfigError, load_scene_config

__all__ = ["SceneConfig", "SceneConfigError", "load_scene_config"]

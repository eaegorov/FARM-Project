"""Public captioning API, loaded only when caption execution is requested.

Importing pure evidence/geometry helpers must not initialize inference libraries.
"""

from importlib import import_module

_EXPORTS = {
    "CaptionManager": "services",
    "CaptionWorker": "worker",
    "ObjectCaptionTask": "models",
    "ObjectCaptionResult": "models",
    "StructuredCaption": "models",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

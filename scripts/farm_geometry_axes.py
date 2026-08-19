"""Shared, fail-closed scene up-direction helpers for FARM geometry stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
INDEX_AXIS = {value: key for key, value in AXIS_INDEX.items()}


@dataclass(frozen=True)
class UpPolicy:
    vector: np.ndarray
    source: str
    axis: str | None
    direction: str

    @property
    def legacy_state_axis(self) -> str:
        return self.axis if self.axis is not None else "vector"

    @property
    def axis_index(self) -> int:
        return AXIS_INDEX[self.axis] if self.axis is not None else 1

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "axis": self.axis,
            "direction": self.direction,
            "vector": self.vector.astype(float).tolist(),
            "axis_aligned": self.axis is not None,
        }


def normalize_up_vector(value: str | Sequence[float] | np.ndarray) -> np.ndarray:
    """Parse and normalize an exact world-space up vector."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError("--up-vector must contain exactly three comma-separated numbers")
        try:
            vector = np.asarray([float(part) for part in parts], dtype=np.float64)
        except ValueError as exc:
            raise ValueError("--up-vector contains a non-numeric value") from exc
    else:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size != 3:
            raise ValueError("up vector must contain exactly three values")
    if not np.isfinite(vector).all():
        raise ValueError("up vector must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("up vector must be non-zero")
    return vector / norm


def resolve_up_policy(
    *,
    up_axis: str = "y",
    up_direction: str = "positive",
    up_vector: str | Sequence[float] | np.ndarray | None = None,
) -> UpPolicy:
    """Resolve vector priority while retaining explicit axis metadata."""

    axis = str(up_axis).lower()
    direction = str(up_direction).lower()
    if axis not in AXIS_INDEX:
        raise ValueError(f"up_axis must be one of {tuple(AXIS_INDEX)}")
    if direction not in {"positive", "negative"}:
        raise ValueError("up_direction must be positive or negative")
    if up_vector is None:
        vector = np.zeros(3, dtype=np.float64)
        vector[AXIS_INDEX[axis]] = 1.0 if direction == "positive" else -1.0
        return UpPolicy(vector=vector, source="axis", axis=axis, direction=direction)

    vector = normalize_up_vector(up_vector)
    dominant = int(np.argmax(np.abs(vector)))
    residual = np.delete(vector, dominant)
    if abs(abs(float(vector[dominant])) - 1.0) <= 1.0e-8 and float(np.linalg.norm(residual)) <= 1.0e-8:
        resolved_axis: str | None = INDEX_AXIS[dominant]
        resolved_direction = "positive" if vector[dominant] > 0.0 else "negative"
    else:
        resolved_axis = None
        resolved_direction = "custom"
    return UpPolicy(
        vector=vector,
        source="vector",
        axis=resolved_axis,
        direction=resolved_direction,
    )


def horizontal_plane_basis(up_vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal 3x2 basis normal to exact up."""

    up = normalize_up_vector(up_vector)
    reference_index = int(np.argmin(np.abs(up)))
    first = np.zeros(3, dtype=np.float64)
    first[reference_index] = 1.0
    first -= up * float(np.dot(first, up))
    first /= max(float(np.linalg.norm(first)), 1.0e-12)
    second = np.cross(up, first)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    return np.column_stack((first, second))


def add_up_arguments(parser: argparse.ArgumentParser, *, default_axis: str = "y") -> None:
    parser.add_argument(
        "--up-axis", choices=("x", "y", "z"), default=default_axis,
        help="Legacy axis-aligned world up; ignored when --up-vector is supplied.",
    )
    parser.add_argument(
        "--up-direction", choices=("positive", "negative"), default="positive",
        help="Sign for --up-axis. Use --up-vector for an oblique direction.",
    )
    parser.add_argument(
        "--up-vector", default=None, metavar="X,Y,Z",
        help="Exact world up vector; takes priority over --up-axis/--up-direction.",
    )


def policy_from_args(args: argparse.Namespace) -> UpPolicy:
    return resolve_up_policy(
        up_axis=str(args.up_axis),
        up_direction=str(args.up_direction),
        up_vector=args.up_vector,
    )


def write_state_up_policy(state: dict, policy: UpPolicy) -> None:
    """Persist exact up while making oblique vectors fail in axis-only readers."""

    state["object_geometry_up_axis"] = policy.legacy_state_axis
    state["object_geometry_up_vector"] = policy.vector.astype(float).tolist()
    state["object_geometry_up_direction"] = policy.direction
    state["object_geometry_up_source"] = policy.source

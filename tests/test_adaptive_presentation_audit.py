from __future__ import annotations

import numpy as np

from scripts.evaluation.audit_farm_adaptive_presentation import inside_rate, route_object


def test_inside_rate_uses_final_oriented_box() -> None:
    angle = np.deg2rad(90.0)
    wxyz = np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    points = np.asarray([[0.0, 0.9, 0.0], [0.0, 1.1, 0.0], [0.4, 0.0, 0.0]])
    assert inside_rate(points, np.zeros(3), np.asarray([2.0, 1.0, 1.0]), wxyz) == 2.0 / 3.0


def test_routing_promotes_low_gaussian_containment_to_targeted_sam3() -> None:
    route, reasons, _ = route_object(
        voxel_inside=0.82,
        gaussian_inside=0.71,
        heldout_status="verified",
        heldout_iou=0.72,
        heldout_precision=0.85,
        projected_iou=0.61,
        box_support=0.91,
        deformable=False,
    )
    assert route == "full_colmap_sam3"
    assert "gaussian_inside_below_hard_threshold" in reasons


def test_routing_leaves_strong_object_and_refits_borderline_object() -> None:
    strong = route_object(
        voxel_inside=0.91,
        gaussian_inside=0.89,
        heldout_status="verified",
        heldout_iou=0.70,
        heldout_precision=0.82,
        projected_iou=0.58,
        box_support=0.90,
        deformable=False,
    )
    assert strong[0] == "leave"
    borderline = route_object(
        voxel_inside=0.79,
        gaussian_inside=None,
        heldout_status="unavailable",
        heldout_iou=None,
        heldout_precision=None,
        projected_iou=0.46,
        box_support=0.82,
        deformable=False,
    )
    assert borderline[0] == "cheap_refit"

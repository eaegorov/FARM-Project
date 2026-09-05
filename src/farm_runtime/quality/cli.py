"""Lazy CLI routing: lightweight help does not initialize inference models."""

import argparse
import importlib
import sys

ACTIONS = {
    "scene-vocabulary": "scene_vocabulary",
    "concept-discovery": "concept_discovery",
    "proposal-geometry": "proposal_geometry",
    "scope-evidence": "scope_evidence",
    "boxer": "boxer",
    "surface-obb": "surface_obb",
    "refinement": "refinement",
    "discovery": "discovery",
    "discovery-review": "discovery_review",
    "export-review": "export_review",
    "camera-refinement": "camera_refinement",
    "lift-ablation": "lift_ablation",
    "lift-review": "lift_review",
    "alignment-review": "alignment_review",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ACTIONS:
        module = importlib.import_module("farm_runtime.quality." + ACTIONS[argv[0]])
        return module.main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="farm quality",
        description="Measured FARM quality development; all commands use new output directories.",
    )
    parser.add_argument("action", choices=sorted(ACTIONS))
    parser.parse_args(argv)
    return 0

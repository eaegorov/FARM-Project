"""Importing an optional feature backend must not enable GPU autotuning."""

import subprocess
import sys


def test_dino_import_preserves_cudnn_algorithm_search_policy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, torch; "
            "torch.backends.cudnn.benchmark = False; "
            "from scene_graph.segmentation import dino; "
            "assert torch.backends.cudnn.benchmark is False; "
            "torch.backends.cudnn.benchmark = True; "
            "importlib.reload(dino); "
            "assert torch.backends.cudnn.benchmark is True",
        ],
        check=True,
    )

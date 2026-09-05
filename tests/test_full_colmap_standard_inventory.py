from farm_runtime.standard import STANDARD_STAGE_CODE_INPUTS


def test_full_colmap_fold_builder_is_in_signed_part_whole_inventory() -> None:
    inputs = set(STANDARD_STAGE_CODE_INPUTS["part_whole"])
    assert {
        "scripts/evaluation/build_farm_full_colmap_folds.py",
        "src/farm_runtime/assembly_mask_reconstruction.py",
        "src/farm_runtime/full_colmap_folds.py",
        "src/farm_runtime/full_colmap_view_planner.py",
        "src/farm_runtime/source_state_canonicalization.py",
        "src/farm_runtime/source_view_canonicalization.py",
    } <= inputs

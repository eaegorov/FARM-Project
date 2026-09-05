import cv2
import numpy as np

from farm_runtime.camera_alignment import propose_camera_pose, refine_camera_rig


def fixture():
    rng = np.random.default_rng(23)
    points = rng.uniform([-3, -3, 4], [3, 3, 8], (600, 3))
    K = np.array([[450.0, 0, 448], [0, 450.0, 448], [0, 0, 1.0]])
    delta = np.eye(4)
    delta[:3, :3] = cv2.Rodrigues(np.array([0.012, -0.025, 0.009]))[0]
    delta[:3, 3] = [0.02, -0.01, 0.03]
    observations = []
    for x in (0.0, 0.0295):
        initial = np.eye(4)
        initial[0, 3] = x
        true = delta @ initial
        camera = (points - true[:3, 3]) @ true[:3, :3]
        pixels = camera @ K.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        pixels += rng.normal(0, 0.1, pixels.shape)
        pixels[:35] = rng.uniform(100, 790, (35, 2))
        observations.append(
            dict(
                points_world_m=points,
                pixels=pixels,
                K=K,
                T_world_cam=initial,
                shape_hw=(896, 896),
            )
        )
    return observations, delta


def test_pnp_validates_on_separate_spatial_points_and_recovers_metric_pose():
    obs, true = fixture()
    o = obs[0]
    pose, report = propose_camera_pose(
        o["points_world_m"], o["pixels"], o["K"], o["T_world_cam"], o["shape_hw"]
    )
    assert report["accepted"] and report["validation_after_px"] < 0.3
    assert np.linalg.norm(pose[:3, 3] - true[:3, 3]) < 0.005


def test_joint_refinement_preserves_rig_baseline_and_relative_rotations():
    obs, true = fixture()
    poses, report = refine_camera_rig(obs)
    assert report["accepted"] and report["validation_after_px"] < 0.4
    expected = np.linalg.inv(obs[0]["T_world_cam"]) @ obs[1]["T_world_cam"]
    assert np.allclose(np.linalg.inv(poses[0]) @ poses[1], expected, atol=1e-12)
    assert abs(np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3]) - 0.0295) < 1e-12
    assert np.linalg.norm(poses[0][:3, 3] - true[:3, 3]) < 0.01


def test_insufficient_evidence_preserves_original_camera():
    obs, _ = fixture()
    o = obs[0]
    o["pixels"] = o["pixels"][:5]
    o["points_world_m"] = o["points_world_m"][:5]
    poses, report = refine_camera_rig([o])
    assert not report["accepted"]
    assert np.array_equal(poses[0], o["T_world_cam"])

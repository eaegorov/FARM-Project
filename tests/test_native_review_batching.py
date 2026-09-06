from types import SimpleNamespace

import numpy as np
from PIL import Image

from farm_runtime.quality import native_observations as native
from farm_runtime.quality_baseline import describe_file


def test_reverse_review_batches_channels_and_preserves_timestamp_and_output_order(
    tmp_path, monkeypatch
):
    frames = []
    records = []
    for image_id, timestamp in enumerate(("a", "a", "b")):
        path = tmp_path / f"source{image_id}.png"
        Image.new("RGB", (32, 32), (40, 70, 100)).save(path)
        frames.append(
            SimpleNamespace(
                image_id=image_id,
                physical_timestamp=timestamp,
                depth_size=(32, 32),
                source_image=path.name,
            )
        )
        records.append(
            dict(
                image_id=image_id,
                source=describe_file(path),
                applied_quarter_turns=image_id,
            )
        )
    run = SimpleNamespace(
        objects=[
            SimpleNamespace(
                object_id=i,
                observations=[SimpleNamespace(image_id=j) for j in (0, 1, 2)],
            )
            for i in range(18)
        ],
        frame=lambda i: frames[i],
    )
    bank = dict(object_ids=np.arange(17), indices=np.arange(17), indptr=np.arange(18))
    render_calls = []
    decode_calls = []
    panels = []

    def render(gaussians, run, frame, ids, members):
        render_calls.append((frame.image_id, list(ids)))
        mass = np.zeros((32, 32, len(ids)), np.float32)
        for column, oid in enumerate(ids):
            assert members[oid].tolist() == [oid]
            mass[oid, frame.image_id + 3, column] = 1
        return (mass,)

    def masks(run, frame, observations, config):
        decode_calls.append((frame.image_id, list(observations)))
        result = {}
        for oid, rows in observations.items():
            assert len(rows) == 1 and rows[0].image_id == frame.image_id
            raw = np.zeros((32, 32), bool)
            raw[oid, frame.image_id + 1] = True
            result[oid] = dict(raw=raw)
        return result, None, None, None

    rotate = native.rotate_image

    def capture(array, turns):
        panels.append((array.copy(), turns))
        return rotate(array, turns)

    monkeypatch.setattr(native.lift, "reverse_render", render)
    monkeypatch.setattr(native.lift, "_load_view_masks", masks)
    monkeypatch.setattr(native, "rotate_image", capture)
    output = tmp_path / "visuals"
    output.mkdir()
    result = native.reverse_review(
        None,
        run,
        bank,
        {"heldout": {"alpha_threshold": 0.5}},
        dict(frames=records),
        output,
    )
    expected = [
        f"group_{oid:04d}_view_{frame:04d}.jpg" for oid in range(17) for frame in (0, 2)
    ]
    assert [r["path"].rsplit("/", 1)[-1] for r in result] == expected
    assert render_calls == decode_calls
    assert [len(ids) for _, ids in render_calls] == [8, 8, 1, 8, 8, 1]
    assert {frame for frame, _ in render_calls} == {0, 2}
    for index, (frame, oid) in enumerate((f, i) for f in (0, 2) for i in range(17)):
        rgb, turns = panels[3 * index]
        source, source_turns = panels[3 * index + 1]
        predicted, predicted_turns = panels[3 * index + 2]
        assert turns == source_turns == predicted_turns == frame
        assert np.argwhere(np.any(source != rgb, axis=-1)).tolist() == [
            [oid, frame + 1]
        ]
        assert np.argwhere(np.any(predicted != rgb, axis=-1)).tolist() == [
            [oid, frame + 3]
        ]
    # Empty native objects neither decode source masks nor run the renderer.
    previous = len(render_calls)
    assert (
        native.reverse_review(
            None,
            run,
            dict(object_ids=[], indices=np.array([], int), indptr=np.array([0])),
            {"heldout": {"alpha_threshold": 0.5}},
            dict(frames=records),
            output,
        )
        == []
    )
    assert len(render_calls) == previous

#!/usr/bin/env python3
"""Render a bounded YOLOE vocabulary audit on exact RGB-D frame records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scene_graph.segmentation.yoloe import YOLOESegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="yoloe-v8l")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=12)
    return parser.parse_args()


def _palette(index: int) -> tuple[int, int, int]:
    return (
        48 + (67 * index) % 208,
        48 + (131 * index) % 208,
        48 + (193 * index) % 208,
    )


def main() -> None:
    args = parse_args()
    frames_payload = json.loads(args.frames_json.read_text(encoding="utf-8"))
    frame_by_timestamp = {
        int(row["timestamp_ns"]): row for row in frames_payload["frames"]
    }
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    selected = [
        *inventory["folds"]["discovery"],
        *inventory["folds"]["verification"],
    ][: args.max_frames]
    additions = set(map(str, inventory.get("additions") or []))
    segmenter = YOLOESegmenter(
        model_id=args.model_id,
        vocab_file=args.vocabulary,
        imgsz=640,
        conf_thres=args.confidence,
        iou_thres=0.5,
        device=args.device,
        use_dino_features=False,
        max_det=200,
        min_mask_pixels=50,
        min_depth_points=50,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    thumbnails = []
    for frame_index, selected_row in enumerate(selected):
        timestamp = int(selected_row["timestamp_ns"])
        frame = frame_by_timestamp[timestamp]
        rgb_path = (args.frames_json.parent / frame["rgb_path"]).resolve(strict=True)
        depth_path = (args.frames_json.parent / frame["depth_path"]).resolve(strict=True)
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path, allow_pickle=False)
        output = segmenter(
            torch.from_numpy(rgb),
            torch.from_numpy(np.asarray(depth, dtype=np.float32)),
            torch.from_numpy(np.asarray(frame["K"], dtype=np.float32)),
            camera_names=[str(frame.get("camera") or "camera")],
            offline_debug=True,
        )
        overlay = bgr.copy()
        detections = []
        names = [
            str(row.get("Object Category") or "unknown")
            for row in output.get("detections", [])
        ]
        scores = output["scores"].detach().cpu().tolist()
        boxes = output["boxes_xyxy"].detach().cpu().tolist()
        masks = [mask.detach().cpu().numpy() for mask in output["masks"]]
        for det_index, (name, score, box, mask) in enumerate(
            zip(names, scores, boxes, masks)
        ):
            colour = _palette(det_index)
            binary = np.asarray(mask, dtype=bool)
            tint = np.zeros_like(overlay)
            tint[:] = colour
            overlay[binary] = cv2.addWeighted(
                overlay, 0.35, tint, 0.65, 0.0
            )[binary]
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(
                overlay,
                f"{name} {score:.2f}",
                (max(0, x0), max(18, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "name": str(name),
                    "score": float(score),
                    "bbox_xyxy": [float(value) for value in box],
                    "mask_pixels": int(binary.sum()),
                    "mask_fraction": float(binary.mean()),
                    "adaptive_term": str(name) in additions,
                }
            )
        thumb = cv2.resize(overlay, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.putText(
            thumb,
            f"t={timestamp} det={len(detections)} adaptive={sum(d['adaptive_term'] for d in detections)}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
        rows.append(
            {
                "timestamp_ns": timestamp,
                "rgb_path": str(rgb_path),
                "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                "detections": detections,
            }
        )
    columns = 4
    blank = np.zeros_like(thumbnails[0])
    while len(thumbnails) % columns:
        thumbnails.append(blank.copy())
    contact = np.concatenate(
        [
            np.concatenate(thumbnails[index : index + columns], axis=1)
            for index in range(0, len(thumbnails), columns)
        ],
        axis=0,
    )
    cv2.imwrite(
        str(args.output_dir / "detections_contact_sheet.jpg"),
        contact,
        [int(cv2.IMWRITE_JPEG_QUALITY), 94],
    )
    report = {
        "schema": "farm.yoloe-vocabulary-audit.v1",
        "status": "PASS",
        "model_id": args.model_id,
        "confidence": args.confidence,
        "vocabulary": str(args.vocabulary.resolve()),
        "vocabulary_sha256": hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "frame_count": len(rows),
        "detection_count": sum(len(row["detections"]) for row in rows),
        "adaptive_detection_count": sum(
            bool(item["adaptive_term"])
            for row in rows
            for item in row["detections"]
        ),
        "frames": rows,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("frame_count", "detection_count", "adaptive_detection_count")}))


if __name__ == "__main__":
    main()

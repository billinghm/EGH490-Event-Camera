#!/usr/bin/env python3
"""
Run YOLOv8 over a video or a folder of event frames, write an annotated
video and a CSV of detections.

Frame video, native and downscaled to match the event camera:
    python detect.py clip.mp4
    python detect.py clip.mp4 --downscale 320 --grey

Event frames from export_events.py (pass the export folder):
    python detect.py events_w20ms_grey

Video input is centre-cropped to a square first, so field of view is
identical between runs and only resolution changes.
"""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

TARGETS = {"sports ball", "bottle"}


def open_source(path):
    """Yield (frame, time_s) from a video file or an export_events.py folder."""
    p = Path(path)
    if p.is_dir():
        index = p / "frames.csv"
        if not index.exists():
            raise SystemExit(f"{p} has no frames.csv - is it an export_events.py folder?")
        with open(index, newline="") as f:
            rows = list(csv.DictReader(f))
        t0 = int(rows[0]["t_start_us"]) if rows else 0
        fps = 1e6 / (int(rows[0]["t_end_us"]) - int(rows[0]["t_start_us"])) if rows else 50.0

        def gen():
            for r in rows:
                img = cv2.imread(str(p / "frames" / r["file"]), cv2.IMREAD_COLOR)
                yield img, (int(r["t_start_us"]) - t0) / 1e6
        return gen(), fps

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise SystemExit(f"could not open {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, i / fps
            i += 1
        cap.release()
    return gen(), fps


def prep(frame, size, grey):
    """Centre-crop to a square, then optionally downscale and greyscale."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    frame = frame[y0:y0 + side, x0:x0 + side]

    if size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    if grey:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)  # YOLO wants 3 channels
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="yolov8n/s/m/l/x.pt - downloads on first use")
    ap.add_argument("--downscale", type=int, default=0,
                    help="resize the square crop to NxN, e.g. 320")
    ap.add_argument("--grey", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--all-classes", action="store_true",
                    help="log every COCO class, not just ball and bottle")
    args = ap.parse_args()

    tag = f"{args.downscale or 'native'}{'_grey' if args.grey else ''}"
    stem = Path(args.video).stem
    out_video = f"{stem}_{tag}.mp4"
    out_csv = f"{stem}_{tag}.csv"

    model = YOLO(args.model)
    source, fps = open_source(args.video)

    writer = None
    rows = []
    frame_idx = 0

    for frame, t in source:
        frame = prep(frame, args.downscale, args.grey)

        res = model.predict(frame, conf=args.conf, verbose=False)[0]

        for box in res.boxes:
            name = model.names[int(box.cls)]
            if not args.all_classes and name not in TARGETS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "frame": frame_idx,
                "time_s": round(t, 4),
                "class": name,
                "conf": round(float(box.conf), 4),
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            })

        annotated = res.plot()
        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(annotated)
        frame_idx += 1

    if writer:
        writer.release()

    fields = ["frame", "time_s", "class", "conf", "x1", "y1", "x2", "y2"]
    with open(out_csv, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        cw.writerows(rows)

    hit_frames = len({r["frame"] for r in rows})
    print(f"\nframes processed:       {frame_idx}")
    print(f"frames with a target:   {hit_frames} "
          f"({100 * hit_frames / max(frame_idx, 1):.1f}%)")
    for name in sorted({r["class"] for r in rows}):
        n = sum(1 for r in rows if r["class"] == name)
        mean = sum(r["conf"] for r in rows if r["class"] == name) / n
        print(f"  {name:<14} {n:>5} detections, mean conf {mean:.3f}")
    print(f"\nwrote {out_video}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

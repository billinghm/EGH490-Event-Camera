#!/usr/bin/env python3
"""
Convert a Prophesee .raw (or .hdf5) recording into event frames.

Each frame covers one fixed time window. Output per run:
    <out>/frames/000000.png ...   lossless 8-bit greyscale, native sensor size
    <out>/frames.csv              window times and event counts per frame

Usage:
    python src/export_events.py path/to/events.raw
    python src/export_events.py events.raw --window-us 10000 --out runs/w10
    python src/export_events.py events.raw --start-s 5 --duration-s 10

Frame modes:
    grey   mid-grey background, ON events white, OFF events black (default)
    count  events per pixel, scaled so the busiest pixel is white
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from metavision_core.event_io import EventsIterator


def make_frame(evs, height, width, mode):
    """Build one 8-bit frame from a structured array of events (x, y, p, t)."""
    if mode == "grey":
        frame = np.full((height, width), 128, dtype=np.uint8)
        if len(evs):
            # Events are time-ordered, so later events at a pixel overwrite
            # earlier ones: each pixel shows its most recent polarity.
            frame[evs["y"], evs["x"]] = np.where(evs["p"] > 0, 255, 0)
        return frame

    if mode == "count":
        counts = np.zeros((height, width), dtype=np.uint32)
        if len(evs):
            np.add.at(counts, (evs["y"], evs["x"]), 1)
        peak = counts.max()
        if peak == 0:
            return np.zeros((height, width), dtype=np.uint8)
        return (counts * (255.0 / peak)).astype(np.uint8)

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", help=".raw or .hdf5 file")
    ap.add_argument("--out", help="output folder (default: next to recording)")
    ap.add_argument("--window-us", type=int, default=20000,
                    help="accumulation window in microseconds (default 20000)")
    ap.add_argument("--mode", choices=["grey", "count"], default="grey")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--duration-s", type=float, default=0.0,
                    help="0 means the whole recording")
    args = ap.parse_args()

    rec = Path(args.recording)
    out = Path(args.out) if args.out else rec.parent / f"{rec.stem}_w{args.window_us // 1000}ms_{args.mode}"
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(input_path=str(rec), mode="delta_t", delta_t=args.window_us,
                  start_ts=int(args.start_s * 1e6))
    if args.duration_s > 0:
        kwargs["max_duration"] = int(args.duration_s * 1e6)
    it = EventsIterator(**kwargs)
    height, width = it.get_size()

    print(f"recording : {rec}")
    print(f"sensor    : {width} x {height}")
    print(f"window    : {args.window_us} us ({1e6 / args.window_us:.1f} frames/s), mode {args.mode}")
    print(f"output    : {out}")

    t0 = time.perf_counter()
    n_frames = 0
    n_total = 0
    n_invalid_total = 0

    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "file", "t_start_us", "t_end_us",
                    "n_events", "n_on", "n_off", "event_rate_per_s", "n_invalid"])

        for evs in it:
            t_end = it.get_current_time()
            t_start = t_end - args.window_us

            # Drop events whose coordinates fall outside the sensor. These come
            # from the RAW decoder, not the scene; count them so it's visible.
            valid = (evs["x"] < width) & (evs["y"] < height)
            n_invalid = int(len(evs) - valid.sum())
            if n_invalid:
                evs = evs[valid]
                n_invalid_total += n_invalid

            frame = make_frame(evs, height, width, args.mode)
            name = f"{n_frames:06d}.png"
            cv2.imwrite(str(frames_dir / name), frame)

            n = len(evs)
            n_on = int((evs["p"] > 0).sum()) if n else 0
            w.writerow([n_frames, name, t_start, t_end, n, n_on, n - n_on,
                        round(n * 1e6 / args.window_us, 1), n_invalid])

            n_frames += 1
            n_total += n
            if n_frames % 250 == 0:
                print(f"  {n_frames} frames, {n_total:,} events")

    elapsed = time.perf_counter() - t0
    span_s = n_frames * args.window_us / 1e6
    print(f"\nframes    : {n_frames} covering {span_s:.1f} s of recording")
    print(f"events    : {n_total:,} (mean {n_total / max(span_s, 1e-9):,.0f} per s)")
    print(f"took      : {elapsed:.1f} s ({elapsed / max(span_s, 1e-9):.2f} s per recorded s)")
    if n_invalid_total:
        pct = 100 * n_invalid_total / (n_total + n_invalid_total)
        bad_frames = "see n_invalid column in frames.csv"
        print(f"WARNING   : dropped {n_invalid_total:,} out-of-range events ({pct:.3f}%) - {bad_frames}")


if __name__ == "__main__":
    main()
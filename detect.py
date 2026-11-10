#!/usr/bin/env python3
"""
Live drone detection: laptop webcam today, Pi Camera Module 3 by Friday.

Laptop:   python detect.py --weights yolo11n.pt
Pi:       python detect.py --source picam --weights best_ncnn_model --headless --save out.mp4 --infinity

Every detection is logged as one JSON line (shaped like the planned VisualDetection message),
so the log carries straight into the M7/M10 work later.
"""
import argparse
import datetime as dt
import json
import time
import uuid
from pathlib import Path

import cv2
from ultralytics import YOLO

# Stock COCO weights have no "drone" class. A drone usually shows up as one of these, if at all.
COCO_PROXY = {"airplane", "bird", "kite"}


def open_source(kind, index, width, height, infinity):
    if kind == "picam":
        from picamera2 import Picamera2

        cam = Picamera2()
        # "RGB888" in picamera2 comes out in BGR byte order, which is what OpenCV expects.
        cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
        cam.start()
        if infinity:
            from libcamera import controls

            cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})
        return (lambda: (True, cam.capture_array())), cam.stop

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {index}. Try --cam 1")
    return cap.read, cap.release


def pick_targets(names, override):
    if override:
        return {n.strip().lower() for n in override.split(",")}
    drone_like = {n.lower() for n in names.values() if any(k in n.lower() for k in ("drone", "uav", "quad"))}
    return drone_like or COCO_PROXY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "picam"], default="webcam")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--weights", default="yolo11n.pt", help=".pt file or exported *_ncnn_model dir")
    ap.add_argument("--imgsz", type=int, default=640, help="try 320 on the Pi for speed")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--targets", default="", help="comma list of class names to alert on; default auto")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--infinity", action="store_true", help="Pi cam: lock focus at infinity")
    ap.add_argument("--headless", action="store_true", help="no window (Pi over SSH)")
    ap.add_argument("--save", default="", help="write annotated video, e.g. out.mp4")
    ap.add_argument("--log", default="detections.jsonl")
    args = ap.parse_args()

    model = YOLO(args.weights, task="detect")
    targets = pick_targets(model.names, args.targets)
    print(f"Model classes of interest: {sorted(targets)}")

    read, close = open_source(args.source, args.cam, args.width, args.height, args.infinity)
    session_id = str(uuid.uuid4())
    log = open(args.log, "a")
    writer = None
    fps, t_prev = 0.0, time.monotonic()
    frames = hits = 0

    try:
        while True:
            ok, frame = read()
            if not ok:
                break
            t_cap = time.monotonic()
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            latency_ms = (time.monotonic() - t_cap) * 1000

            drone_in_frame = False
            for box in res.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                is_target = label.lower() in targets
                color = (0, 0, 255) if is_target else (160, 160, 160)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if is_target:
                    drone_in_frame = True
                    log.write(json.dumps({
                        "session_id": session_id,
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "mono_s": round(t_cap, 4),
                        "label": label,
                        "confidence": round(conf, 3),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "image_wh": [frame.shape[1], frame.shape[0]],
                        "latency_ms": round(latency_ms, 1),
                        "camera_pose_ref": None,  # v2 gimbal hook
                    }) + "\n")

            frames += 1
            hits += drone_in_frame
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            banner = "DRONE IN FRAME" if drone_in_frame else "clear"
            cv2.putText(frame, f"{banner} | {fps:.1f} fps | {latency_ms:.0f} ms", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if drone_in_frame else (0, 200, 0), 2)

            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(frame)

            if args.headless:
                if frames % 30 == 0:
                    print(f"{frames} frames | {fps:.1f} fps | {latency_ms:.0f} ms | drone frames: {hits}")
            else:
                cv2.imshow("drone detect (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        close()
        log.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Done: {frames} frames, {hits} with drone, avg ~{fps:.1f} fps. Log: {args.log}")


if __name__ == "__main__":
    main()
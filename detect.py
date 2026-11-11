#!/usr/bin/env python3
"""
Live drone detection: laptop webcam or Pi Camera Module 3.

Pi (no PyTorch needed, runs the NCNN model directly):
    python detect.py --source picam --weights best_ncnn_model --imgsz 320 --headless --save out.mp4
Smoke test on one photo:
    python detect.py --weights best_ncnn_model --imgsz 320 --image test.jpg
Laptop (uses Ultralytics for .pt weights):
    python detect.py --weights best.pt

If --weights is an exported *_ncnn_model folder, inference uses the `ncnn` package plus numpy/OpenCV
only. Ultralytics/PyTorch are imported only for .pt weights. Every detection is logged as one JSON
line (shaped like the planned VisualDetection message).
"""
import argparse
import datetime as dt
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

# Stock COCO weights have no "drone" class. A drone usually shows up as one of these, if at all.
COCO_PROXY = {"airplane", "bird", "kite"}


# ---------------------------------------------------------------- detectors
# Both return a list of (class_id, confidence, x1, y1, x2, y2) in original-frame pixels.

class NcnnDetector:
    """Runs an Ultralytics-exported NCNN model (model.ncnn.param/.bin + metadata.yaml) without PyTorch."""

    def __init__(self, folder, imgsz, threads=4, iou=0.5):
        import ncnn
        import yaml

        folder = Path(folder)
        meta = yaml.safe_load((folder / "metadata.yaml").read_text())
        self.names = {int(k): str(v) for k, v in meta["names"].items()}
        self.end2end = bool(meta.get("end2end", False))
        exported = meta.get("imgsz", [imgsz, imgsz])
        exported = exported[0] if isinstance(exported, (list, tuple)) else int(exported)
        if imgsz != exported:
            print(f"Note: model was exported at imgsz {exported}; using {exported} (re-export to change).")
        self.imgsz = int(exported)
        self.iou = iou

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        self.net.load_param(str(folder / "model.ncnn.param"))
        self.net.load_model(str(folder / "model.ncnn.bin"))
        self.in_name = self.net.input_names()[0]
        self.out_name = self.net.output_names()[0]

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def detect(self, frame, conf):
        img, r, left, top = self._letterbox(frame)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0  # BGR->RGB, CHW

        ex = self.net.create_extractor()
        ex.input(self.in_name, self.ncnn.Mat(x).clone())
        _, out = ex.extract(self.out_name)
        out = np.array(out).squeeze()

        h, w = frame.shape[:2]
        dets = []

        if self.end2end:  # NMS-free heads (e.g. YOLO26): rows of x1,y1,x2,y2,score,cls
            if out.ndim == 2 and out.shape[0] < out.shape[1]:
                out = out.T
            for x1, y1, x2, y2, s, c in out[:, :6]:
                if s >= conf:
                    dets.append((int(c), float(s), *self._unmap(x1, y1, x2, y2, r, left, top, w, h)))
            return dets

        # Standard YOLOv8/11 head: (4 + num_classes, num_anchors), boxes as cx,cy,w,h in input pixels.
        if out.shape[0] > out.shape[1]:
            out = out.T
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= conf
        if not keep.any():
            return dets
        boxes, best, cls = boxes[keep], best[keep], cls[keep]

        xywh = np.column_stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 2], boxes[:, 3]])
        offset = cls[:, None].astype(np.float32) * 4096.0  # class-aware NMS: shift each class apart
        shifted = np.column_stack([xywh[:, :2] + offset, xywh[:, 2:]])
        idx = cv2.dnn.NMSBoxes(shifted.tolist(), best.tolist(), conf, self.iou)
        for i in np.array(idx).flatten():
            bx, by, bw, bh = xywh[i]
            dets.append((int(cls[i]), float(best[i]), *self._unmap(bx, by, bx + bw, by + bh, r, left, top, w, h)))
        return dets

    @staticmethod
    def _unmap(x1, y1, x2, y2, r, left, top, w, h):
        x1 = int(np.clip((x1 - left) / r, 0, w - 1))
        x2 = int(np.clip((x2 - left) / r, 0, w - 1))
        y1 = int(np.clip((y1 - top) / r, 0, h - 1))
        y2 = int(np.clip((y2 - top) / r, 0, h - 1))
        return x1, y1, x2, y2


class UltralyticsDetector:
    """For .pt weights on the laptop. Needs ultralytics + PyTorch."""

    def __init__(self, weights, imgsz):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.names = self.model.names
        self.imgsz = imgsz

    def detect(self, frame, conf):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf, verbose=False)[0]
        return [(int(b.cls), float(b.conf), *map(int, b.xyxy[0].tolist())) for b in res.boxes]


def load_detector(weights, imgsz):
    if Path(weights).is_dir() and (Path(weights) / "model.ncnn.param").exists():
        return NcnnDetector(weights, imgsz)
    return UltralyticsDetector(weights, imgsz)


# ---------------------------------------------------------------- camera + helpers

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


def draw(frame, dets, names, targets):
    """Draws boxes; returns the target detections as (label, conf, box)."""
    hits = []
    for c, conf, x1, y1, x2, y2 in dets:
        label = names.get(c, str(c))
        is_target = label.lower() in targets
        color = (0, 0, 255) if is_target else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_target:
            hits.append((label, conf, [x1, y1, x2, y2]))
    return hits


def run_image(det, path, conf, targets):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read {path}")
    det.detect(frame, conf)  # warm-up run
    times = []
    for _ in range(5):
        t = time.monotonic()
        dets = det.detect(frame, conf)
        times.append((time.monotonic() - t) * 1000)
    hits = draw(frame, dets, det.names, targets)
    out = str(Path(path).with_name(Path(path).stem + "_det.jpg"))
    cv2.imwrite(out, frame)
    for c, s, *box in dets:
        print(f"  {det.names.get(c, c)} {s:.2f} {box}")
    print(f"{len(dets)} detections ({len(hits)} target) | median {sorted(times)[2]:.0f} ms/frame "
          f"(~{1000 / sorted(times)[2]:.1f} fps) | saved {out}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "picam"], default="webcam")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--weights", default="yolo11n.pt", help=".pt file or exported *_ncnn_model folder")
    ap.add_argument("--imgsz", type=int, default=640, help="for NCNN, the size the model was exported at")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--targets", default="", help="comma list of class names to alert on; default auto")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--infinity", action="store_true", help="Pi cam: lock focus at infinity")
    ap.add_argument("--headless", action="store_true", help="no window (Pi over SSH)")
    ap.add_argument("--save", default="", help="write annotated video, e.g. out.mp4")
    ap.add_argument("--log", default="detections.jsonl")
    ap.add_argument("--image", default="", help="run once on a photo, print timing, save *_det.jpg, exit")
    args = ap.parse_args()

    det = load_detector(args.weights, args.imgsz)
    targets = pick_targets(det.names, args.targets)
    print(f"Backend: {type(det).__name__} | imgsz {det.imgsz} | classes of interest: {sorted(targets)}")

    if args.image:
        run_image(det, args.image, args.conf, targets)
        return

    read, close = open_source(args.source, args.cam, args.width, args.height, args.infinity)
    session_id = str(uuid.uuid4())
    log = open(args.log, "a")
    writer = None
    fps, t_prev = 0.0, time.monotonic()
    frames = hits_total = 0

    try:
        while True:
            ok, frame = read()
            if not ok:
                break
            t_cap = time.monotonic()
            dets = det.detect(frame, args.conf)
            latency_ms = (time.monotonic() - t_cap) * 1000

            hits = draw(frame, dets, det.names, targets)
            for label, conf, box in hits:
                log.write(json.dumps({
                    "session_id": session_id,
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "mono_s": round(t_cap, 4),
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": box,
                    "image_wh": [frame.shape[1], frame.shape[0]],
                    "latency_ms": round(latency_ms, 1),
                    "camera_pose_ref": None,  # v2 gimbal hook
                }) + "\n")

            drone_in_frame = bool(hits)
            frames += 1
            hits_total += drone_in_frame
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
                    print(f"{frames} frames | {fps:.1f} fps | {latency_ms:.0f} ms | drone frames: {hits_total}")
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
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Done: {frames} frames, {hits_total} with drone, avg ~{fps:.1f} fps. Log: {args.log}")


if __name__ == "__main__":
    main()
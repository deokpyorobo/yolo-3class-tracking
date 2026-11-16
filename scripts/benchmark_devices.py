"""best.pt 를 OpenVINO 로 변환해 CPU / 내장 GPU(Arc) / NPU(Intel AI Boost) 추론 속도를 비교한다.

사용법:
    python scripts/benchmark_devices.py --source input/clip.webm --start 3 --duration 5
결과: results/device_benchmark.json
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def frames(src, start, dur):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    out = []
    for _ in range(int(round(dur * fps))):
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def bench(model, imgs, **kw):
    for f in imgs[:5]:  # warm-up (NPU/GPU 컴파일 포함)
        model.predict(f, verbose=False, **kw)
    t, n_det = time.perf_counter(), 0
    for f in imgs:
        n_det += len(model.predict(f, verbose=False, **kw)[0].boxes)
    ms = (time.perf_counter() - t) / len(imgs) * 1000
    return {"ms_per_frame": round(ms, 1), "fps": round(1000 / ms, 1), "detections": n_det}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    imgs = frames(args.source, args.start, args.duration)
    ov_dir = Path(YOLO(args.weights).export(format="openvino", imgsz=640, half=False, dynamic=False))

    res = {"frames": len(imgs), "imgsz": 640, "torch_threads": args.threads}
    res["cpu_pytorch"] = bench(YOLO(args.weights), imgs, conf=args.conf, device="cpu")
    for dev in ("cpu", "gpu", "npu"):
        try:
            res[f"openvino_{dev}"] = bench(YOLO(str(ov_dir), task="detect"), imgs, conf=args.conf,
                                           device=dev if dev == "npu" else f"intel:{dev}")
        except Exception as e:  # noqa: BLE001
            res[f"openvino_{dev}"] = {"error": str(e)[:200]}
        print(dev, res[f"openvino_{dev}"])
    out = ROOT / "results" / "device_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

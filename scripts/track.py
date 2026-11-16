"""학습된 모델로 동영상에서 cup/bottle/laptop 을 감지·추적하고 결과 영상을 만든다.

- model.track(persist=True) 로 프레임마다 ByteTrack 추적 (ID 유지)
- 각 객체 ID 의 중심점 궤적(trajectory)을 영상에 선으로 그리고 CSV 로 따로 저장
- 부류별 고유 ID 수(카운팅)를 화면에 표시
- 궤적 Plot(PNG)과 README 미리보기용 GIF 도 함께 저장

사용법:
    python scripts/track.py --source input.webm --start 12 --duration 5
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
# 부류별 궤적 색 (BGR): cup=blue #2a78d6, bottle=aqua #1baf7a, laptop=orange #eb6834
COLORS = {0: (214, 120, 42), 1: (122, 175, 27), 2: (52, 104, 235)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--start", type=float, default=0.0, help="시작 시각(초)")
    ap.add_argument("--duration", type=float, default=5.0, help="길이(초)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--min-len", type=int, default=5, help="카운트에 포함할 최소 추적 프레임 수")
    ap.add_argument("--fade", type=int, default=10, help="사라진 객체 궤적을 유지할 프레임 수")
    ap.add_argument("--out", default=str(ROOT / "results" / "tracking_result.mp4"))
    ap.add_argument("--threads", type=int, default=2, help="CPU 스레드 수 (노트북 발열 억제)")
    ap.add_argument("--device", default=None,
                    help="cpu, 0(CUDA) 또는 OpenVINO 모델일 때 npu / intel:gpu (예: --weights models/best_openvino_model --device npu)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights, task="detect")
    names = model.names

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    n_frames = int(round(args.duration * fps))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trails = defaultdict(list)  # track id -> [(x, y)]
    id_cls = {}
    last_seen = {}  # track id -> 마지막으로 감지된 프레임
    rows = []
    gif_frames = []

    for f in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, conf=args.conf, tracker=args.tracker, device=args.device,
                        verbose=False)[0]
        vis = r.plot(line_width=2)
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                     r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                trails[tid].append((cx, cy))
                id_cls[tid] = c
                last_seen[tid] = f
                rows.append([f, round(f / fps, 3), tid, names[c], round(p, 3), round(cx, 1), round(cy, 1),
                             round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        # 궤적 그리기: 최근 args.fade 프레임 안에 보인 객체만, 최근 45개 점
        for tid, pts in trails.items():
            if f - last_seen[tid] <= args.fade and len(pts) > 1:
                cv2.polylines(vis, [np.int32(pts[-45:])], False, COLORS[id_cls[tid]], 3)
        # 부류별 누적 고유 ID 카운트 (min_len 프레임 이상 추적된 ID 만 → 순간적인 오검출 제외)
        counts = defaultdict(int)
        for tid, c in id_cls.items():
            if len(trails[tid]) >= args.min_len:
                counts[c] += 1
        text = "  ".join(f"{names[k]}: {counts[k]}" for k in sorted(names))
        cv2.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(vis, f"t={f / fps:4.1f}s  IDs  {text}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)
        if f % 3 == 0:
            g = cv2.resize(vis, (480, int(480 * H / W)))
            gif_frames.append(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)))
    writer.release()
    cap.release()

    with open(out.with_name("trajectories.csv"), "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["frame", "time_s", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    trails = {t: p for t, p in trails.items() if len(p) >= args.min_len}
    id_cls = {t: c for t, c in id_cls.items() if t in trails}
    for tid, pts in trails.items():
        p = np.array(pts)
        col = np.array(COLORS[id_cls[tid]][::-1]) / 255
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=2)
        ax.plot(p[-1, 0], p[-1, 1], "o", color=col)
        ax.annotate(f"{names[id_cls[tid]]} #{tid}", p[-1], fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title("Object center trajectories")
    fig.tight_layout()
    fig.savefig(out.with_name("trajectories.png"))

    if gif_frames:
        gif_frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif_frames[1:],
                           duration=int(3000 / fps), loop=0)
    summary = {names[k]: sum(c == k for c in id_cls.values()) for k in sorted(names)}
    print(f"저장: {out} ({len(rows)} detections, {len(trails)} tracks) 부류별 ID 수: {summary}")


if __name__ == "__main__":
    main()

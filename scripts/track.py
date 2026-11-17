"""학습된 모델로 동영상 또는 웹캠에서 cup/bottle/laptop 을 감지·추적한다.

- model.track(persist=True) 로 프레임마다 ByteTrack 추적 (ID 유지)
- 각 객체 ID 의 중심점 궤적(trajectory)을 영상에 선으로 그리고 CSV 로 따로 저장
- 부류별 고유 ID 수(카운팅)를 화면에 표시
- 궤적 Plot(PNG)과 README 미리보기용 GIF 도 함께 저장

사용법:
    # 동영상 파일 (5초 구간)
    python scripts/track.py --source input/source_clip_3s-8s.mp4 --duration 5
    # 웹캠 실시간 (q / ESC 로 종료, 화면에 FPS 표시, 결과는 results/live/ 에 녹화)
    python scripts/track.py --source 0
    # Jetson CSI 카메라 + TensorRT 엔진, 모니터 없이 30초만 녹화
    python scripts/track.py --source csi --weights models/best.engine --device 0 --no-show --duration 30
    # 동영상 파일을 카메라처럼 실시간 재생하며 추적 (데모용)
    python scripts/track.py --source input/source_clip_3s-8s.mp4 --live
"""
import argparse
import csv
import sys
import time
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
MAX_TRAIL = 2000  # 실시간 모드에서 ID 당 저장할 최대 궤적 점 수 (메모리 제한)
MAX_GIF = 150  # GIF 최대 프레임 수 (실시간 모드에서 앞부분만 저장)


def csi_pipeline(w, h, fps, sensor_id=0, flip=0):
    """Jetson CSI 카메라(IMX219 등)용 GStreamer 파이프라인 (OpenCV 가 GStreamer 지원으로 빌드돼 있어야 함)."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width={w}, height={h}, "
            f"framerate={fps}/1 ! nvvidconv flip-method={flip} ! video/x-raw, width={w}, height={h}, "
            f"format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1 max-buffers=1")


class GstPipeCapture:
    """OpenCV 가 GStreamer 없이 빌드된 경우(pip opencv) gst-launch-1.0 의 출력을 직접 읽는 CSI 카메라 입력.
    cv2.VideoCapture 와 같은 read() / isOpened() / get() / set() / release() 를 제공한다."""

    def __init__(self, pipeline, w, h, fps):
        import shlex
        import subprocess

        self.w, self.h, self.fps = w, h, fps
        self.size = w * h * 3
        cmd = ["gst-launch-1.0", "-q"] + shlex.split(pipeline.replace("appsink drop=1 max-buffers=1",
                                                                      "fdsink fd=1 sync=false"))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=self.size)

    def isOpened(self):
        return self.proc.poll() is None

    def read(self):
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return False, None
        return True, np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3).copy()

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.w,
                cv2.CAP_PROP_FRAME_HEIGHT: self.h}.get(prop, 0)

    def set(self, *_):
        return False

    def release(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def open_source(args):
    """(VideoCapture, is_camera) 반환. 숫자 → USB 웹캠, 'csi' → Jetson CSI, rtsp/http → 스트림."""
    src = args.source
    if src.isdigit():
        cap = None
        if sys.platform == "win32":  # Windows 는 DSHOW 가 보통 더 빨리 열림, 안 되면 기본 백엔드(MSMF)
            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 오래된 프레임이 쌓여 지연되는 것 방지
        return cap, True
    if src.lower() == "csi":
        pipe = csi_pipeline(args.cam_width, args.cam_height, 30)
        import re

        if re.search(r"GStreamer:\s*YES", cv2.getBuildInformation()):
            return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER), True
        return GstPipeCapture(pipe, args.cam_width, args.cam_height, 30), True  # pip opencv 등 GStreamer 미지원
    if src.lower().startswith(("rtsp://", "http://", "https://")):
        return cv2.VideoCapture(src), True
    return cv2.VideoCapture(src), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="동영상 경로, 웹캠 번호(0), csi, 또는 rtsp:// URL")
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--start", type=float, default=0.0, help="시작 시각(초, 동영상 파일만)")
    ap.add_argument("--duration", type=float, default=None,
                    help="길이(초). 기본: 동영상 5초, 실시간은 제한 없음(q 로 종료)")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--min-len", type=int, default=5, help="카운트에 포함할 최소 추적 프레임 수")
    ap.add_argument("--fade", type=int, default=10, help="사라진 객체 궤적을 유지할 프레임 수")
    ap.add_argument("--threads", type=int, default=2, help="CPU 스레드 수 (노트북 발열 억제)")
    ap.add_argument("--device", default=None,
                    help="cpu, 0(CUDA/Jetson) 또는 OpenVINO 모델일 때 npu / intel:gpu")
    ap.add_argument("--live", action="store_true", help="동영상 파일도 실시간 모드로 처리 (화면 표시, 벽시계 기준)")
    # (Python 3.8 호환: BooleanOptionalAction 대신 --show / --no-show 를 직접 정의)
    ap.add_argument("--show", dest="show", action="store_true", default=None,
                    help="화면 표시 (기본: 실시간 모드에서만 표시)")
    ap.add_argument("--no-show", dest="show", action="store_false", help="화면 표시 끔 (모니터 없을 때)")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--out", default=None, help="결과 영상 경로 (기본: results/tracking_result.mp4, "
                                                "실시간은 results/live/tracking_live_<시각>.mp4)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cap, is_camera = open_source(args)
    if not cap.isOpened():
        sys.exit(f"입력을 열 수 없습니다: {args.source}")
    live = is_camera or args.live
    show = live if args.show is None else args.show
    duration = args.duration if args.duration is not None else (0 if live else 5.0)  # 0 = 제한 없음

    if args.out:
        out = Path(args.out)
    elif live:
        out = ROOT / "results" / "live" / f"tracking_live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    else:
        out = ROOT / "results" / "tracking_result.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights, task="detect")
    names = model.names

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30
    if not is_camera and args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * src_fps))
    max_frames = int(round(duration * src_fps)) if (duration and not live) else None

    trails = defaultdict(list)  # track id -> [(x, y)]
    id_cls = {}
    last_seen = {}  # track id -> 마지막으로 감지된 프레임
    gif_frames = []
    writer = None
    pending = []  # 실시간 모드: 녹화 fps 를 재기 위해 앞 프레임을 잠시 보관
    fps_ema = None
    window = "YOLO tracking (q / ESC to quit)"

    # 동영상: 결과 영상과 같은 폴더의 trajectories.csv / 실시간: <영상이름>_trajectories.csv (녹화마다 따로)
    csv_path = out.with_name(f"{out.stem}_trajectories.csv" if live else "trajectories.csv")
    csv_fp = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_fp)
    csv_w.writerow(["frame", "time_s", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])

    t0 = time.perf_counter()
    f = 0
    try:
        while True:
            if max_frames is not None and f >= max_frames:
                break
            if live and duration and time.perf_counter() - t0 >= duration:
                break
            t_frame = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if is_camera:
                    print("카메라에서 프레임을 받지 못했습니다.")
                break
            H, W = frame.shape[:2]
            t_now = (time.perf_counter() - t0) if live else f / src_fps

            r = model.track(frame, persist=True, conf=args.conf, tracker=args.tracker, device=args.device,
                            verbose=False)[0]
            vis = r.plot(line_width=2)
            if r.boxes.id is not None:
                for (x1, y1, x2, y2), tid, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                         r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    trails[tid].append((cx, cy))
                    if len(trails[tid]) > MAX_TRAIL:
                        del trails[tid][0]
                    id_cls[tid] = c
                    last_seen[tid] = f
                    csv_w.writerow([f, round(t_now, 3), tid, names[c], round(p, 3), round(cx, 1), round(cy, 1),
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
            dt = time.perf_counter() - t_frame
            fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / dt
            head = f"t={t_now:4.1f}s  " + (f"{fps_ema:4.1f} FPS  " if live else "") + f"IDs  {text}"
            cv2.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
            cv2.putText(vis, head, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # 녹화: 파일은 원본 fps, 실시간은 처음 30프레임의 실제 처리 속도로 fps 를 정함
            if writer is None:
                if not live:
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
                else:
                    pending.append(vis)
                    if len(pending) >= 30:
                        rec_fps = max(1.0, min(src_fps, len(pending) / (time.perf_counter() - t0)))
                        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (W, H))
                        for p in pending:
                            writer.write(p)
                        pending.clear()
            else:
                writer.write(vis)
            if f % 3 == 0 and len(gif_frames) < MAX_GIF:
                g = cv2.resize(vis, (480, int(480 * H / W)))
                gif_frames.append(Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB)))

            if show:
                try:
                    cv2.imshow(window, vis)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    print("화면을 열 수 없어(모니터 없음) 표시 없이 녹화만 합니다.")
                    show = False
            f += 1
    except KeyboardInterrupt:
        print("중단됨 (Ctrl+C) — 지금까지의 결과를 저장합니다.")
    finally:
        cap.release()
        csv_fp.close()
        if writer is None and pending:  # 30프레임 전에 끝난 실시간 녹화
            rec_fps = max(1.0, min(src_fps, len(pending) / max(time.perf_counter() - t0, 1e-3)))
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), rec_fps,
                                     (pending[0].shape[1], pending[0].shape[0]))
            for p in pending:
                writer.write(p)
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if f == 0:
        sys.exit("처리한 프레임이 없습니다.")
    elapsed = time.perf_counter() - t0

    trails = {t: p for t, p in trails.items() if len(p) >= args.min_len}
    id_cls = {t: c for t, c in id_cls.items() if t in trails}
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
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
    fig.savefig(csv_path.with_suffix(".png"))

    if gif_frames:
        gif_frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=gif_frames[1:],
                           duration=int(3000 / src_fps), loop=0)
    summary = {names[k]: sum(c == k for c in id_cls.values()) for k in sorted(names)}
    print(f"저장: {out} ({f} frames, {elapsed:.1f}s, 평균 {f / elapsed:.1f} FPS, {len(trails)} tracks) "
          f"부류별 ID 수: {summary}")


if __name__ == "__main__":
    main()

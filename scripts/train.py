"""YOLOv8n 을 cup/bottle/laptop 3개 부류 데이터셋으로 미세조정(fine-tuning)한다.

기본값은 2차(최종) 실험 설정이고, --v1 을 주면 1차(기본 설정) 실험을 재현한다.
(실제 학습은 같은 설정으로 colab/train_colab.ipynb 에서 T4 GPU 로 진행)

사용법:
    python scripts/train.py --device 0          # 2차(최종)
    python scripts/train.py --device 0 --v1     # 1차
결과: runs/detect/<name>/ (weights/best.pt, results.png, confusion_matrix.png 등)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

V1 = dict(epochs=50, patience=15)  # 나머지는 Ultralytics 기본값
V2 = dict(epochs=100, patience=30, freeze=10, optimizer="AdamW", lr0=0.001,
          warmup_epochs=1, warmup_bias_lr=0.0, cos_lr=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true", help="1차(기본 설정) 실험 재현")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    name = "cup_bottle_laptop" if args.v1 else "cup_bottle_laptop_v2"
    model = YOLO("yolov8n.pt")  # COCO 사전학습 가중치에서 시작
    model.train(
        data=str(ROOT / "datasets" / "cup_bottle_laptop" / "data.yaml"),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        seed=0,
        project=str(ROOT / "runs" / "detect"),
        name=name,
        exist_ok=True,
        plots=True,
        **(V1 if args.v1 else V2),
    )
    metrics = model.val()
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")

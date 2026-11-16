"""Colab 학습 결과(runs/colab/...)에서 보고서용 그림·지표를 results/training 으로 모으고
1차(v1) vs 2차(v2) 비교 그래프를 만든다."""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"v1": ROOT / "runs/colab/cup_bottle_laptop", "v2": ROOT / "runs/colab/cup_bottle_laptop_v2"}
OUT = ROOT / "results" / "training"
KEEP = ["results.png", "results.csv", "args.yaml", "final_metrics.json", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "BoxF1_curve.png", "labels.jpg", "train_batch0.jpg", "val_batch0_pred.jpg",
        "val_batch0_labels.jpg"]

# 팔레트: 범주형 slot1 blue / slot2 orange, 잉크·축은 중립색
C = {"v1": "#2a78d6", "v2": "#eb6834"}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK})

for k, d in RUNS.items():
    (OUT / k).mkdir(parents=True, exist_ok=True)
    for f in KEEP:
        if (d / f).exists():
            shutil.copy2(d / f, OUT / k / f)

df = {k: pd.read_csv(d / "results.csv") for k, d in RUNS.items()}
met = {k: json.loads((d / "final_metrics.json").read_text()) for k, d in RUNS.items()}
label = {"v1": "v1 (default)", "v2": "v2 (freeze=10, no bias warm-up)"}

# 1) epoch 별 검증 mAP — small multiples (같은 y 스케일)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, col, title in zip(axes, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], ["Validation mAP50", "Validation mAP50-95"]):
    for k in RUNS:
        ax.plot(df[k]["epoch"], df[k][col], color=C[k], lw=2, label=label[k])
        b = df[k][col].idxmax()
        ax.plot(df[k]["epoch"][b], df[k][col][b], "o", ms=8, color=C[k], mec=SURF, mew=2)
        ax.annotate(f"{k} best {df[k][col][b]:.3f} @ep{int(df[k]['epoch'][b])}", (df[k]["epoch"][b], df[k][col][b]),
                    textcoords="offset points", xytext=(6, 8), fontsize=9, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel("epoch")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_ylim(0, 0.85)
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "v1_vs_v2_map_per_epoch.png", dpi=130)

# 2) 부류별 최종 mAP50 (best.pt) — 묶음 막대
classes = list(met["v2"]["per_class"])
fig, ax = plt.subplots(figsize=(7, 3.6))
w = 0.38
for i, k in enumerate(RUNS):
    vals = [met[k]["per_class"][c]["mAP50"] for c in classes] + [met[k]["mAP50"]]
    xs = [j + (i - 0.5) * (w + 0.02) for j in range(len(vals))]
    ax.bar(xs, vals, width=w, color=C[k], label=label[k], edgecolor=SURF, linewidth=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(range(len(classes) + 1), classes + ["all"])
ax.set_ylim(0, 1)
ax.set_title("Final validation mAP50 by class (best.pt)", loc="left", color=INK, fontsize=11)
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "v1_vs_v2_map_by_class.png", dpi=130)

summary = {k: {"epochs_run": int(df[k]["epoch"].iloc[-1]), "train_time_s": round(float(df[k]["time"].iloc[-1]), 1),
               "best_epoch": int(df[k]["metrics/mAP50-95(B)"].idxmax() + 1), **{m: round(met[k][m], 4)
               for m in ["precision", "recall", "mAP50", "mAP50-95"]},
               "per_class_mAP50": {c: round(met[k]["per_class"][c]["mAP50"], 4) for c in classes}}
           for k in RUNS}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

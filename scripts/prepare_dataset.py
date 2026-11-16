"""COCO 2017 에서 3개 부류(cup, bottle, laptop) 이미지를 수집해 YOLO 형식 데이터셋으로 변환한다.

- 부류별로 해당 객체가 '충분히 크게' 보이는 이미지를 N장씩 고른다 (기본 200장).
- 선택된 이미지 안의 3개 부류 객체는 모두 라벨링한다 (iscrowd 제외).
- train/val = 8:2 로 분할하고 data.yaml 을 생성한다.

사용법:
    python scripts/prepare_dataset.py --per-class 200
"""
import argparse
import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CLASSES = ["cup", "bottle", "laptop"]  # YOLO class id 0, 1, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--min-area", type=float, default=0.02, help="이미지 대비 최소 객체 면적 비율")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    cache = ROOT / "datasets" / "_coco"
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "annotations_trainval2017.zip"
    jpath = cache / "instances_train2017.json"
    if not jpath.exists():
        if not zpath.exists():
            print("COCO annotation 다운로드 중 (~240MB)...")
            urlretrieve(ANN_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            jpath.write_bytes(z.read("annotations/instances_train2017.json"))

    print("annotation 로드 중...")
    coco = json.loads(jpath.read_text())
    name2cid = {c["name"]: c["id"] for c in coco["categories"]}
    cid2yolo = {name2cid[n]: i for i, n in enumerate(CLASSES)}
    images = {im["id"]: im for im in coco["images"]}

    anns = defaultdict(list)  # image_id -> [(yolo_cls, bbox)]
    big = defaultdict(set)  # yolo_cls -> image_ids where that class is large enough
    for a in coco["annotations"]:
        if a["category_id"] not in cid2yolo or a["iscrowd"]:
            continue
        im = images[a["image_id"]]
        k = cid2yolo[a["category_id"]]
        anns[a["image_id"]].append((k, a["bbox"]))
        if a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"]) >= args.min_area:
            big[k].add(a["image_id"])

    chosen = {}  # image_id -> class it was chosen for
    for k in range(len(CLASSES)):
        pool = sorted(big[k] - chosen.keys())
        random.shuffle(pool)
        for iid in pool[: args.per_class]:
            chosen[iid] = k
    print("부류별 선택:", {CLASSES[k]: sum(v == k for v in chosen.values()) for k in range(len(CLASSES))})

    # 부류별 비율을 유지하며 train/val 분할
    split = {}
    for k in range(len(CLASSES)):
        ids = [i for i, v in chosen.items() if v == k]
        random.shuffle(ids)
        n_val = int(len(ids) * args.val_ratio)
        split.update({i: "val" for i in ids[:n_val]})
        split.update({i: "train" for i in ids[n_val:]})

    out = ROOT / "datasets" / "cup_bottle_laptop"
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    inst = defaultdict(int)
    for iid, s in split.items():
        im = images[iid]
        W, H = im["width"], im["height"]
        lines = []
        for k, (x, y, w, h) in anns[iid]:
            if w < 1 or h < 1:
                continue
            lines.append(f"{k} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            inst[(s, CLASSES[k])] += 1
        stem = Path(im["file_name"]).stem
        (out / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    def fetch(iid):
        im = images[iid]
        dst = out / "images" / split[iid] / im["file_name"]
        if not dst.exists():
            urlretrieve(im["coco_url"], dst)

    print(f"이미지 {len(split)}장 다운로드 중...")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, split))

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    n_tr = sum(v == "train" for v in split.values())
    print(f"완료: train {n_tr}장 / val {len(split) - n_tr}장")
    print("객체 수:", dict(sorted(inst.items())))
    json.dump({"images": {"train": n_tr, "val": len(split) - n_tr},
               "instances": {f"{s}/{c}": n for (s, c), n in sorted(inst.items())}},
              open(out / "stats.json", "w"), indent=2)


if __name__ == "__main__":
    main()

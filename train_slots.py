"""
Train the parking-slot detector on the Kaggle "Parking Space Detection Based on Top-View" dataset.

Put the dataset in  dataset/  next to this file (any inner folder layout works), then run:
    python train_slots.py

It reads YOLO .txt labels (boxes or polygons) or Pascal-VOC .xml labels, keeps your
train/val split if the folders have one (otherwise splits 85/15), builds yolo_data/,
trains, and saves the best weights to models/slots.pt (app.py uses it automatically).
"""
import shutil, random, json
from collections import Counter
import xml.etree.ElementTree as ET
from pathlib import Path
import torch, yaml
from PIL import Image
from ultralytics import YOLO

SRC, OUT = Path("dataset"), Path("yolo_data")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GPU = torch.cuda.is_available()
EPOCHS, IMGSZ = (80, 1024) if GPU else (30, 640)   # CPU is slow: train on a free Colab/Kaggle GPU if you can
BASE = "yolov8s.pt"                                  # yolov8m.pt = more accurate, needs more memory


def read_yolo(p):
    rows = []
    for line in p.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        try:
            v = [float(x) for x in t]
        except ValueError:
            return None
        if len(t) == 5:
            rows.append((t[0], *v[1:]))
        else:  # polygon / oriented box -> bounding box
            xs, ys = v[1::2], v[2::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            rows.append((t[0], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return rows


def read_voc(p):
    root = ET.parse(p).getroot()
    W, H = float(root.findtext("size/width", 0)), float(root.findtext("size/height", 0))
    if not W or not H:
        return None
    rows = []
    for o in root.iter("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        rows.append((o.findtext("name").strip(), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return rows


def read_coco(p):
    """One big COCO-style .json holding the boxes of many images."""
    d = json.loads(p.read_text())
    if not isinstance(d, dict) or "images" not in d or "annotations" not in d:
        return {}
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    imgs = {i["id"]: i for i in d["images"]}
    out = {}
    for a in d["annotations"]:
        i = imgs.get(a["image_id"])
        if not i or not i.get("width") or not i.get("height"):
            continue
        x, y, w, h = a["bbox"]
        W, H = i["width"], i["height"]
        out.setdefault(Path(i["file_name"]).stem, []).append(
            (str(cats.get(a["category_id"], a["category_id"])), (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
    return out


def read_json_label(p, img):
    """One .json per image: LabelMe / X-AnyLabeling ('shapes'), Supervisely ('objects'), or a 'bbox' list."""
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {"annotations": d}
    W = d.get("imageWidth") or (d.get("size") or {}).get("width")
    H = d.get("imageHeight") or (d.get("size") or {}).get("height")
    if not W or not H:
        with Image.open(img) as im:
            W, H = im.size
    items = []
    for lb in d.get("labels") or []:                     # {"labels":[{"name","x1","y1","x2","y2"}]}
        if isinstance(lb, dict) and "x1" in lb:
            items.append((str(lb.get("name", "parking_space")), [(lb["x1"], lb["y1"]), (lb["x2"], lb["y2"])]))
    for sh in d.get("shapes") or []:
        items.append((str(sh.get("label", "parking_space")), sh.get("points", [])))
    for ob in d.get("objects") or []:
        items.append((str(ob.get("classTitle", "parking_space")), (ob.get("points") or {}).get("exterior", [])))
    for a in d.get("annotations") or []:
        if isinstance(a, dict) and "bbox" in a:
            x, y, w, h = a["bbox"]
            items.append((str(a.get("label", a.get("category_name", "parking_space"))), [(x, y), (x + w, y + h)]))
    rows = []
    for lab, pts in items:
        if len(pts) < 2:
            continue
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        rows.append((lab, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return rows


def split_of(p):
    parts = {x.lower() for x in p.parts}
    if parts & {"val", "valid", "validation", "test"}:
        return "val"
    return "train" if "train" in parts else None


def main():
    if not SRC.exists():
        raise SystemExit("Folder 'dataset' not found next to train_slots.py")
    labels = {}
    for p in SRC.rglob("*"):
        if p.suffix.lower() in (".txt", ".xml") and p.name.lower() not in ("classes.txt", "readme.txt"):
            labels.setdefault(p.stem, p)
            labels.setdefault(Path(p.stem).stem, p)          # names like "img.jpg.xml"
    coco = {}
    for j in SRC.rglob("*.json"):
        try:
            coco.update(read_coco(j))
        except Exception:
            pass
    images = [p for p in SRC.rglob("*") if p.suffix.lower() in IMG_EXT]
    parsed = []
    jsons = {p.stem: p for p in SRC.rglob("*.json")}
    for img in images:
        rows = coco.get(img.stem)
        if not rows and img.stem in jsons:
            try:
                rows = read_json_label(jsons[img.stem], img)
            except Exception:
                rows = None
        lab = labels.get(img.stem)
        if rows is None and lab is not None:
            rows = read_voc(lab) if lab.suffix.lower() == ".xml" else read_yolo(lab)
        if rows:
            parsed.append((img, rows))
    print(f"Found {len(images)} images, {len(parsed)} with usable labels.")
    if not parsed:
        files = [p for p in SRC.rglob("*") if p.is_file()]
        print("File types in dataset/:", dict(Counter(p.suffix.lower() for p in files)))
        print("Example files:", *[str(p) for p in files[:8]], sep="\n  ")
        js = next((p for p in files if p.suffix.lower() == ".json"), None)
        if js:
            print("First JSON file starts with:\n", js.read_text(encoding="utf-8", errors="ignore")[:500])
        raise SystemExit("No image+label pairs found. Send the lines above to get the format supported.")

    id_names = {}
    for y in SRC.rglob("*.yaml"):
        try:
            n = yaml.safe_load(y.read_text()).get("names")
        except Exception:
            continue
        if n:
            id_names = dict(enumerate(n)) if isinstance(n, list) else {int(k): v for k, v in n.items()}
            break

    keys = sorted({r[0] for _, rows in parsed for r in rows},
                  key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))
    cmap = {k: i for i, k in enumerate(keys)}

    def cname(k):
        if k.isdigit() and int(k) in id_names:
            return str(id_names[int(k)])
        if k.isdigit():
            return "parking_space" if len(keys) == 1 else f"class_{k}"
        return k

    random.seed(0)
    buckets = {"train": [], "val": [], None: []}
    for item in parsed:
        buckets[split_of(item[0])].append(item)
    random.shuffle(buckets[None])
    k = int(len(buckets[None]) * 0.85)
    train, val = buckets["train"] + buckets[None][:k], buckets["val"] + buckets[None][k:]
    if not val:
        random.shuffle(train)
        n = max(1, len(train) // 10)
        val, train = train[:n], train[n:]

    if OUT.exists():
        shutil.rmtree(OUT)
    for split, items in (("train", train), ("val", val)):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)
        for n, (img, rows) in enumerate(items):
            stem = f"{n:05d}"
            shutil.copy(img, OUT / split / "images" / (stem + img.suffix.lower()))
            text = "\n".join(f"{cmap[r[0]]} " + " ".join(f"{min(max(v, 0), 1):.6f}" for v in r[1:]) for r in rows)
            (OUT / split / "labels" / (stem + ".txt")).write_text(text)
    names = {i: cname(k) for i, k in enumerate(keys)}
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT.resolve()), "train": "train/images", "val": "val/images", "names": names}))
    print(f"Train {len(train)} | Val {len(val)} | Classes {names}")

    model = YOLO(BASE)
    model.train(data=str(OUT / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, device=0 if GPU else "cpu",
                batch=8 if GPU else 2, workers=2, patience=20, flipud=0.5,
                project="runs", name="slots", exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, "models/slots.pt")
    print("Saved models/slots.pt. Restart app.py to use it.")


if __name__ == "__main__":
    main()
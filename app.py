"""
Vehicle Parking Slot Detection
Upload image -> detect + count vehicles -> detect parking slots -> list blank slots.
Every upload, annotated result and count is stored (uploads/, results/, parking.db).

Run:  python app.py   then open http://127.0.0.1:5000
"""
import os, json, sqlite3, uuid, datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

# ---------------- config ----------------
BASE = os.path.dirname(os.path.abspath(__file__))
UP, RES, MOD = (os.path.join(BASE, d) for d in ("uploads", "results", "models"))
DB = os.path.join(BASE, "parking.db")
for d in (UP, RES, MOD):
    os.makedirs(d, exist_ok=True)

VEHICLE_WEIGHTS = "yolov8m.pt"                 # use yolov8x.pt for more accuracy (slower)
SLOT_WEIGHTS = os.path.join(MOD, "slots.pt")   # optional: your trained slot model
VEHICLE_CLASSES = [2, 3, 5, 7]                 # COCO: car, motorcycle, bus, truck
VEHICLE_CONF = 0.15
TEXTURE_THRESH = 0.18                          # texture fallback: lower = more slots marked occupied
OVERLAP = 0.30                                 # share of a slot covered by a vehicle = occupied

vehicle_model = YOLO(VEHICLE_WEIGHTS)
slot_model = YOLO(SLOT_WEIGHTS) if os.path.exists(SLOT_WEIGHTS) else None
SLOT_CONF = 0.25
try:   # run the slot model at the same image size it was trained with
    SLOT_IMGSZ = int(slot_model.ckpt["train_args"]["imgsz"])
except Exception:
    SLOT_IMGSZ = 1024
# True if the slot model has empty/occupied classes; False if it only detects "parking space"
SLOT_HAS_STATE = bool(slot_model) and any(
    k in str(n).lower() for n in slot_model.names.values()
    for k in ("empty", "free", "vacant", "occupied", "taken", "available"))

app = Flask(__name__)


# ---------------- storage ----------------
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.execute("""CREATE TABLE IF NOT EXISTS runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, image TEXT, result TEXT,
        vehicles INTEGER, total INTEGER, occupied INTEGER, empty INTEGER,
        empty_ids TEXT, mode TEXT)""")
    c.execute("CREATE TABLE IF NOT EXISTS layout(id INTEGER PRIMARY KEY CHECK(id=1), slots TEXT)")


# ---------------- detection ----------------
def _predict(img, off=(0, 0), imgsz=1280):
    r = vehicle_model.predict(img, conf=VEHICLE_CONF, imgsz=imgsz,
                              classes=VEHICLE_CLASSES, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append([x1 + off[0], y1 + off[1], x2 + off[0], y2 + off[1], float(b.conf[0])])
    return out


def detect_vehicles(img, tiled=False):
    """Full-image pass; optional 4 overlapping tiles so small cars in big lots are found."""
    h, w = img.shape[:2]
    boxes = _predict(img)
    if tiled:   # 640px tiles, each enlarged 2x, so small top-view cars become detectable
        size, ov = 640, 128
        def starts(n):
            return [0] if n <= size else list(range(0, n - size, size - ov)) + [n - size]
        for oy in starts(h):
            for ox in starts(w):
                boxes += _predict(img[oy:oy + size, ox:ox + size], (ox, oy), 1280)
    if not boxes:
        return []
    xywh = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [b[4] for b in boxes], VEHICLE_CONF, 0.45)
    return [boxes[i] for i in np.array(keep).flatten()]


def slots_from_layout(img, vehicles, polys):
    """Slots you drew in the browser (normalised coords). Occupied = vehicle box overlaps slot."""
    h, w = img.shape[:2]
    res = []
    for i, p in enumerate(polys):
        pts = (np.array(p) * [w, h]).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        sm = np.zeros((bh, bw), np.uint8)
        cv2.fillPoly(sm, [(pts - np.array([x, y])).astype(np.int32)], 1)
        area, occ = max(int(sm.sum()), 1), False
        for x1, y1, x2, y2, _ in vehicles:
            if x2 < x or x1 > x + bw or y2 < y or y1 > y + bh:
                continue
            vm = np.zeros((bh, bw), np.uint8)
            cv2.rectangle(vm, (int(x1 - x), int(y1 - y)), (int(x2 - x), int(y2 - y)), 1, -1)
            if (sm & vm).sum() / area > OVERLAP:
                occ = True
                break
        res.append({"id": i + 1, "pts": pts, "occupied": occ})
    return res


def slots_from_model(img, vehicles, conf=SLOT_CONF):
    """Slots found by your trained model. If the model only detects 'parking space'
    (no empty/occupied classes), occupancy is decided from vehicle overlap."""
    r = slot_model.predict(img, conf=conf, iou=0.6, imgsz=SLOT_IMGSZ, max_det=500, verbose=False)[0]
    found = []
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        name = r.names[int(b.cls[0])].lower()
        empty = any(k in name for k in ("empty", "free", "vacant", "available", "unoccupied"))
        found.append((x1, y1, x2, y2, not empty))
    found.sort(key=lambda s: (s[1] // 50, s[0]))
    if not SLOT_HAS_STATE:
        h, w = img.shape[:2]
        polys = [[[x1 / w, y1 / h], [x2 / w, y1 / h], [x2 / w, y2 / h], [x1 / w, y2 / h]]
                 for x1, y1, x2, y2, _ in found]
        return slots_from_layout(img, vehicles, polys)
    return [{"id": i + 1, "occupied": o,
             "pts": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)}
            for i, (x1, y1, x2, y2, o) in enumerate(found)]


def texture_occupancy(img, slots):
    """Fallback when no vehicles are detected (common on top-view photos). A slot is 'busy' when
    its inside differs from the bare pavement: different brightness and/or many edges."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 60, 160)
    pave = float(np.median(g))                      # typical pavement brightness of this photo
    out = []
    for s in slots:
        x, y, bw, bh = cv2.boundingRect(s["pts"])
        mx, my = int(bw * 0.2), int(bh * 0.2)      # ignore the painted slot lines
        sl = (slice(y + my, y + bh - my), slice(x + mx, x + bw - mx))
        if g[sl].size == 0:
            out.append(False)
            continue
        dev = float(np.abs(g[sl].astype(np.float32) - pave).mean()) / 255
        edge = float(edges[sl].mean()) / 255
        out.append(dev + 3 * edge > TEXTURE_THRESH)
    return out


def annotate(img, vehicles, slots):
    out = img.copy()
    ov = out.copy()
    for s in slots:
        cv2.fillPoly(ov, [s["pts"]], (60, 60, 230) if s["occupied"] else (80, 200, 60))
    out = cv2.addWeighted(ov, 0.35, out, 0.65, 0)
    fs = max(0.5, img.shape[1] / 1600)
    for s in slots:
        col = (60, 60, 230) if s["occupied"] else (60, 180, 40)
        cv2.polylines(out, [s["pts"]], True, col, 2)
        cx, cy = map(int, s["pts"].mean(axis=0))
        cv2.putText(out, str(s["id"]), (cx - 8, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    for x1, y1, x2, y2, _ in vehicles:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 140, 0), 2)
    return out


# ---------------- routes ----------------
@app.post("/analyze")
def analyze():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded."), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="That file is not a readable image."), 400

    uid = uuid.uuid4().hex[:10]
    name_in, name_out = f"{uid}.jpg", f"{uid}_result.jpg"
    cv2.imwrite(os.path.join(UP, name_in), img)

    row = db().execute("SELECT slots FROM layout WHERE id=1").fetchone()
    polys = json.loads(row["slots"]) if row else []
    # Trained model with Occupied/Vacant classes: it already knows which slots hold a car,
    # so the slower general vehicle-detection pass is skipped.
    use_model_states = bool(slot_model) and SLOT_HAS_STATE and not polys
    vehicles = [] if use_model_states else detect_vehicles(img, request.form.get("tiled") == "1")
    n_vehicles = None
    hint = ""
    if polys:
        slots, mode = slots_from_layout(img, vehicles, polys), "drawn layout"
    elif slot_model:
        conf = float(request.form.get("slot_conf") or SLOT_CONF)
        slots, mode = slots_from_model(img, vehicles, conf), "trained slot model"
        if not slots:
            hint = "The trained model found no slots. Lower 'Slot confidence' (try 0.10) or train for more epochs."
    else:
        slots, mode = [], "no slots defined"
        hint = "models/slots.pt was not found and no layout is saved. Finish train_slots.py, or draw slots by hand."

    if slots and use_model_states:
        n_vehicles = sum(s["occupied"] for s in slots)     # one vehicle per occupied slot
    elif slots and not vehicles:
        for sl, o in zip(slots, texture_occupancy(img, slots)):
            sl["occupied"] = bool(o)
        mode += " + texture check"
        hint = hint or "No vehicles were detected, so occupancy was estimated from image texture. Tick 'High accuracy (tiled)'."
    cv2.imwrite(os.path.join(RES, name_out), annotate(img, vehicles, slots))
    empty_ids = [s["id"] for s in slots if not s["occupied"]]
    stats = dict(vehicles=len(vehicles) if n_vehicles is None else n_vehicles, total=len(slots), occupied=len(slots) - len(empty_ids),
                 empty=len(empty_ids), empty_ids=empty_ids, mode=mode, result=name_out, hint=hint)
    with db() as c:
        c.execute("INSERT INTO runs(ts,image,result,vehicles,total,occupied,empty,empty_ids,mode) "
                  "VALUES(?,?,?,?,?,?,?,?,?)",
                  (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), name_in, name_out,
                   stats["vehicles"], stats["total"], stats["occupied"], stats["empty"],
                   ",".join(map(str, empty_ids)), mode))
    return jsonify(stats)


@app.get("/layout")
def get_layout():
    row = db().execute("SELECT slots FROM layout WHERE id=1").fetchone()
    return jsonify(json.loads(row["slots"]) if row else [])


@app.post("/layout")
def save_layout():
    data = request.get_json(silent=True) or []
    if not all(isinstance(p, list) and len(p) == 4 for p in data):
        return jsonify(error="Each slot needs 4 corner points."), 400
    with db() as c:
        c.execute("INSERT OR REPLACE INTO layout(id,slots) VALUES(1,?)", (json.dumps(data),))
    return jsonify(ok=True, slots=len(data))


@app.get("/history")
def history():
    return jsonify([dict(r) for r in db().execute("SELECT * FROM runs ORDER BY id DESC LIMIT 50")])


@app.get("/files/<kind>/<name>")
def files(kind, name):
    folder = {"uploads": UP, "results": RES}.get(kind)
    return send_from_directory(folder, name) if folder else ("", 404)


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parking Slot Detector</title>
<style>
:root{--asphalt:#1f2326;--panel:#2a2f33;--paint:#f2c230;--free:#3ddc84;--taken:#ff5a4d;--text:#ece8da;--mute:#9aa0a3}
*{box-sizing:border-box}
body{margin:0;background:var(--asphalt);color:var(--text);font:16px/1.5 "Segoe UI",system-ui,sans-serif}
header{padding:20px 24px;border-bottom:4px dashed var(--paint)}
h1{margin:0;font:700 28px/1.1 "Arial Narrow","Segoe UI",sans-serif;letter-spacing:.5px}
header p{margin:4px 0 0;color:var(--mute)}
main{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:20px;padding:20px 24px}
@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border-radius:8px;padding:16px}
h2{margin:0 0 12px;font-size:18px}
canvas,#out{width:100%;height:auto;background:#111;border-radius:6px;display:block}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
button{background:var(--paint);color:#1a1a1a;border:0;border-radius:6px;padding:8px 14px;font-weight:600;cursor:pointer}
button.alt{background:#3a4045;color:var(--text)}
button:focus-visible,input:focus-visible{outline:2px solid var(--paint);outline-offset:2px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}
.stat{background:var(--asphalt);border-radius:6px;padding:10px;text-align:center}
.stat b{display:block;font-size:28px}.stat span{color:var(--mute);font-size:13px}
#msg{color:var(--paint);min-height:24px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:6px 8px;border-bottom:1px solid #3a4045;text-align:left}
td img{width:90px;border-radius:4px;display:block}
.wide{grid-column:1/-1;overflow-x:auto}
</style></head><body>
<header><h1>Parking Slot Detector</h1>
<p>Upload a parking-area photo. Vehicles are counted and blank slots are listed.</p></header>
<main>
<section>
  <h2>1. Image and slots</h2>
  <div class="row">
    <input type="file" id="file" accept="image/*">
    <label><input type="checkbox" id="tiled"> High accuracy (tiled, slower)</label>
    <label>Slot confidence <input type="number" id="sconf" value="0.25" min="0.05" max="0.9" step="0.05" style="width:70px"></label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="drawmode"> Draw slots: click 4 corners per slot</label>
    <button class="alt" id="undo">Undo</button>
    <button class="alt" id="clear">Clear</button>
    <button class="alt" id="save">Save layout</button>
  </div>
  <canvas id="cv" width="800" height="500"></canvas>
  <div class="row" style="margin-top:12px"><button id="go">Detect vehicles and blank slots</button></div>
  <div id="msg"></div>
</section>
<section>
  <h2>2. Result</h2>
  <div class="stats">
    <div class="stat"><b id="vehicles">-</b><span>vehicles</span></div>
    <div class="stat"><b id="total">-</b><span>slots</span></div>
    <div class="stat"><b id="occupied" style="color:var(--taken)">-</b><span>occupied</span></div>
    <div class="stat"><b id="empty" style="color:var(--free)">-</b><span>blank</span></div>
  </div>
  <p>Blank slot numbers: <b id="blank">-</b> <span id="mode" style="color:var(--mute)"></span></p>
  <img id="out" alt="Annotated parking image appears here">
</section>
<section class="wide">
  <h2>Saved uploads</h2>
  <table><thead><tr><th>Result</th><th>Time</th><th>Vehicles</th><th>Slots</th><th>Occupied</th><th>Blank</th><th>Blank slot numbers</th></tr></thead>
  <tbody id="hb"></tbody></table>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d');
let img=null,slots=[],cur=[];

function poly(pts,col,label){
  ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
  ctx.strokeStyle=col;ctx.stroke();ctx.fillStyle=col+'33';ctx.fill();
  ctx.fillStyle='#fff';ctx.font=(cv.width/45)+'px sans-serif';
  ctx.fillText(label,pts[0][0]+4,pts[0][1]+cv.width/45);
}
function draw(){
  if(!img)return;
  ctx.drawImage(img,0,0);
  const W=cv.width,H=cv.height;ctx.lineWidth=Math.max(2,W/400);
  slots.forEach((s,i)=>poly(s.map(p=>[p[0]*W,p[1]*H]),'#f2c230',i+1));
  cur.forEach(p=>{ctx.fillStyle='#ff5a4d';ctx.beginPath();ctx.arc(p[0]*W,p[1]*H,W/200+2,0,7);ctx.fill();});
}
$('file').onchange=()=>{
  const im=new Image();
  im.onload=()=>{img=im;cv.width=im.width;cv.height=im.height;draw();};
  im.src=URL.createObjectURL($('file').files[0]);
};
cv.onclick=e=>{
  if(!$('drawmode').checked||!img)return;
  const r=cv.getBoundingClientRect();
  cur.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);
  if(cur.length==4){slots.push(cur);cur=[];}
  draw();
};
$('undo').onclick=()=>{cur.length?cur.pop():slots.pop();draw();};
$('clear').onclick=()=>{slots=[];cur=[];draw();};
$('save').onclick=async()=>{
  await fetch('/layout',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(slots)});
  $('msg').textContent='Layout saved with '+slots.length+' slots.';
};
$('go').onclick=async()=>{
  const f=$('file').files[0];
  if(!f){$('msg').textContent='Choose an image first.';return;}
  $('msg').textContent='Analyzing...';
  const fd=new FormData();fd.append('image',f);fd.append('tiled',$('tiled').checked?'1':'0');fd.append('slot_conf',$('sconf').value);
  const d=await (await fetch('/analyze',{method:'POST',body:fd})).json();
  if(d.error){$('msg').textContent=d.error;return;}
  ['vehicles','total','occupied','empty'].forEach(k=>$(k).textContent=d[k]);
  $('blank').textContent=d.empty_ids.length?d.empty_ids.join(', '):'none';
  $('mode').textContent='('+d.mode+')';
  $('out').src='/files/results/'+d.result+'?t='+Date.now();
  $('msg').textContent=d.hint||(d.total?'':'No slots defined yet. Tick "Draw slots", click 4 corners per slot, save the layout, then detect again.');
  hist();
};
async function hist(){
  const r=await (await fetch('/history')).json();
  $('hb').innerHTML=r.map(x=>`<tr><td><a href="/files/results/${x.result}" target="_blank"><img src="/files/results/${x.result}" alt=""></a></td>
  <td>${x.ts}</td><td>${x.vehicles}</td><td>${x.total}</td><td>${x.occupied}</td><td>${x.empty}</td><td>${x.empty_ids||'-'}</td></tr>`).join('');
}
(async()=>{slots=await (await fetch('/layout')).json();hist();})();
</script></body></html>"""


if __name__ == "__main__":
    print("Slot mode:", "trained model + drawn layout" if slot_model else "drawn layout only")
    app.run(host="127.0.0.1", port=5000, debug=False)
"""Create a small offline scene/grasp viewer; no hardware, inference, or source edits.

Run: pixi run python -m experiments perception view --help
Dependencies already in the project environment: numpy, Pillow, opencv-python.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_url(path: Path):
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(args):
    root = args.dataset_root / "review/real_v1" / args.scene
    fid = args.frame
    objects = read(root / "scene_meta" / f"{fid}.json")["objects"]
    matches = [k for k, v in objects.items() if v.get("object_name") == args.object]
    if len(matches) != 1:
        raise ValueError(f"Expected one {args.object} instance; found {matches}")
    object_id = matches[0]
    pose = read(root / "6d_pose_json" / f"{fid}.json")
    provenance = read(root / "inference_meta/foundationpose" / args.camera / f"{fid}.json")
    if pose["object_id"] != object_id or provenance.get("camera_name") != args.camera:
        raise ValueError("Object/view mismatch between observation and saved pose")
    if provenance.get("status") != "ok":
        raise ValueError("Saved pose is not marked ok")
    conf = read(root / "conf" / f"{fid}.json")
    camera = next(c for c in conf["cameras"] if c["name"] == args.camera)
    K, T = np.asarray(camera["intrinsic"]), np.asarray(pose["ob_in_cam"])
    rgb_path = root / "rgb" / args.camera / f"{fid}.png"
    with Image.open(rgb_path) as im:
        width, height = im.size
    seg = np.asarray(Image.open(root / "inst_seg" / args.camera / f"{fid}.png"))
    mapping = read(root / "inst_seg" / args.camera / f"semantics_mapping_{fid}.json")
    color = next(ast.literal_eval(k) for k, v in mapping.items() if v.get("class") == object_id)
    mask = np.all(seg == color, axis=-1)
    depth = np.asarray(Image.open(root / "depth" / args.camera / f"{fid}.png"))
    if mask.shape != (height, width) or depth.shape != mask.shape or not mask.any():
        raise ValueError("RGB/depth/mask dimensions disagree, or object mask is empty")
    depth = depth.astype(float) * float(conf["depth"]["scale_to_mm"]) / 1000
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys, xs = np.where(mask)
    crop = [
        max(0, int(xs.min()) - 45),
        max(0, int(ys.min()) - 45),
        min(width, int(xs.max()) + 46),
        min(height, int(ys.max()) + 46),
    ]
    db = args.dataset_root / "isaacsim_grasp_data_gen" / args.object / "UON_3finger_gripper"
    merged = read(db / f"{args.object}_merged_grasp_zero_pose.json")
    if merged["metadata"].get("coordinate_frame") != "object_zero_pose":
        raise ValueError("This viewer expects an object_zero_pose grasp database")
    selected = sorted(merged["data"], key=lambda r: r["score"], reverse=True)[: args.max_grasps]
    if not selected:
        raise ValueError("No grasp candidates")
    raw_cache, rows = {}, []
    for row in selected:
        sid = row["scene_id"]
        if sid not in raw_cache:
            raw_cache[sid] = {r["source_pregrasp_index"]: r for r in read(db / "output_grasp" / f"{sid}.json")}
        raw = raw_cache[sid][row["source_pregrasp_index"]]
        if raw["quality"].get("contact_point_frame") != "object_local":
            raise ValueError("Unexpected contact-point frame")
        g = T @ np.asarray(row["grasp_mat"])
        contacts = [
            {
                "finger": c["sensor"].split("/")[-1],
                "camera": (T[:3, :3] @ np.asarray(c["position"]) + T[:3, 3]).tolist(),
            }
            for c in raw.get("contact_points_object_local", [])
        ]
        rows.append(
            {
                "id": f"{sid}:{row['source_pregrasp_index']}",
                "score": row["score"],
                "origin": g[:3, 3].tolist(),
                "axes": g[:3, :3].T.tolist(),
                "contacts": contacts,
                "object_grasp": row["grasp_mat"],
                "camera_grasp": g.tolist(),
            }
        )
    valid = depth[mask & (depth > 0)]
    overlay_path = root / "diagnostics/foundationpose/overlay" / f"{fid}.png"
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Saved CAD overlay not found: {overlay_path}")
    with Image.open(overlay_path) as im:
        if im.size != (width, height):
            raise ValueError("Saved CAD overlay dimensions differ from RGB")
    return {
        "scene": args.scene,
        "frame": fid,
        "camera": args.camera,
        "object": args.object,
        "object_id": object_id,
        "rgb": image_url(rgb_path),
        "overlay": image_url(overlay_path),
        "width": width,
        "height": height,
        "crop": crop,
        "K": K.tolist(),
        "contours": [c[:, 0, :].tolist() for c in contours],
        "rows": rows,
        "depth_valid_pct": round(float((depth[mask] > 0).mean()) * 100, 1),
        "depth_median_m": round(float(np.median(valid)), 4) if valid.size else None,
        "object_pose": T.tolist(),
        "pose_path": str(root / "6d_pose_json" / f"{fid}.json"),
        "overlay_path": str(overlay_path),
    }


HTML = r"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>단계 1 · 관측과 파지 후보</title><style>
body{font:15px system-ui,sans-serif;margin:24px;background:#f5f7fa;color:#172333;max-width:1250px}
h1{font-size:23px;margin-bottom:8px}p{line-height:1.6}button,select,input{font:inherit}
button,select{padding:7px}label{display:inline-block;margin:6px 16px 6px 0}
.controls,.info{background:white;padding:14px;border-radius:10px;margin:12px 0}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.view{min-width:0}
canvas{display:block;background:#ddd;width:100%;border:1px solid #c6cdd6;aspect-ratio:1.4}
pre{font-size:12px;overflow:auto}#slider{width:230px;vertical-align:middle}
@media(max-width:750px){.views{grid-template-columns:1fr}}
</style><h1>단계 1 · 관측과 파지 후보 연결</h1>
<p id="context"></p><div class="controls">
<button id="prev">이전 후보</button> <select id="candidate"></select> <button id="next">다음 후보</button>
<input id="slider" type="range" min="0" value="0"><br>
<label><input id="zoom" type="checkbox" checked>물체 확대</label>
<label><input id="mask" type="checkbox" checked>마스크 윤곽</label>
<label><input id="origin" type="checkbox" checked>파지 원점</label>
<label><input id="axis" type="checkbox" checked>파지 +Z 방향</label>
<label><input id="contacts" type="checkbox" checked>DB 접촉점</label>
</div><div class="views"><div class="view"><p><b>원본 영상 + 선택한 파지 후보</b></p><canvas id="original"></canvas></div>
<div class="view"><p><b>저장된 CAD 겹침 영상 + 같은 후보</b></p><canvas id="cad"></canvas></div></div>
<div class="info"><div id="details"></div>
<p>노란 선: 물체 마스크 · 검정 원: 파지 좌표 원점 · 분홍 화살표: 파지 +Z 축<br>
주황/청록/보라 점: finger1/2/3의 <b>DB 접촉점</b>. 이 관측에서 실제 접촉했다는 뜻은 아닙니다.</p>
<p>오른쪽 CAD 영상은 기존 FoundationPose 결과입니다. 원점은 실제 TCP로 확정되지 않았으며,
그리퍼 본체는 표시하지 않습니다. 화살표는 이동 경로가 아닌 축 표시입니다.</p>
<details><summary>선택 후보의 좌표·출처 보기</summary><pre id="record"></pre></details></div>
<script>const D=__DATA__;
const $=id=>document.getElementById(id), img=new Image(), overlay=new Image();
let index=0;const colors={finger1:'#ef981b',finger2:'#00b8b3',finger3:'#b05bdd'};
$('context').textContent=`${D.scene} / ${D.camera} / ${D.frame} · ${D.object} (${D.object_id}) · 상위 ${D.rows.length}개 후보`;
for(let i=0;i<D.rows.length;i++){const o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${D.rows[i].id} · score ${D.rows[i].score.toFixed(3)}`;$('candidate').appendChild(o);}
$('slider').max=D.rows.length-1;
function project(p){if(p[2]<=0)return null;return [(D.K[0][0]*p[0]+D.K[0][1]*p[1])/p[2]+D.K[0][2],D.K[1][1]*p[1]/p[2]+D.K[1][2]];}
function draw(id,picture){
 const cv=$(id),c=cv.getContext('2d');cv.width=760;cv.height=540;c.clearRect(0,0,760,540);
 if(!picture.complete||!picture.naturalWidth)return;
 const box=$('zoom').checked?D.crop:[0,0,D.width,D.height],bw=box[2]-box[0],bh=box[3]-box[1];
 const scale=Math.min(cv.width/bw,cv.height/bh),ox=(cv.width-bw*scale)/2,oy=(cv.height-bh*scale)/2;
 c.drawImage(picture,box[0],box[1],bw,bh,ox,oy,bw*scale,bh*scale);
 const xy=p=>[ox+(p[0]-box[0])*scale,oy+(p[1]-box[1])*scale];
 function dot(p,color,r){const uv=project(p);if(!uv)return;const q=xy(uv);c.beginPath();c.arc(q[0],q[1],r,0,2*Math.PI);c.fillStyle=color;c.fill();c.lineWidth=1.5;c.strokeStyle='white';c.stroke();}
 if($('mask').checked){c.strokeStyle='#ffdf21';c.lineWidth=1.5;for(const path of D.contours){c.beginPath();path.forEach((p,i)=>{const q=xy(p);i?c.lineTo(...q):c.moveTo(...q)});c.closePath();c.stroke();}}
 const r=D.rows[index];
 if($('axis').checked){const a=project(r.origin),b=project(r.origin.map((x,i)=>x+.025*r.axes[2][i]));if(a&&b){const p=xy(a),q=xy(b),angle=Math.atan2(q[1]-p[1],q[0]-p[0]);c.strokeStyle='#f630a4';c.lineWidth=3;c.beginPath();c.moveTo(...p);c.lineTo(...q);c.moveTo(q[0]-10*Math.cos(angle-.4),q[1]-10*Math.sin(angle-.4));c.lineTo(...q);c.lineTo(q[0]-10*Math.cos(angle+.4),q[1]-10*Math.sin(angle+.4));c.stroke();}}
 if($('origin').checked)dot(r.origin,'#111',5);
 if($('contacts').checked)for(const p of r.contacts)dot(p.camera,colors[p.finger]||'#aaa',6);
}
function update(){index=Math.max(0,Math.min(D.rows.length-1,index));$('candidate').value=index;$('slider').value=index;
 const r=D.rows[index];$('details').textContent=`후보 ${r.id} · score ${r.score.toFixed(3)} (파지 성공확률 아님) · 접촉 손가락 ${r.contacts.length}개 · 마스크 내 유효 깊이 ${D.depth_valid_pct}% · 깊이 중앙값 ${D.depth_median_m} m`;
 $('record').textContent=JSON.stringify({candidate_id:r.id,grasp_origin_camera_mm:r.origin.map(x=>Math.round(x*100000)/100),approach_camera:r.axes[2],T_camera_grasp:r.camera_grasp,T_object_grasp:r.object_grasp,T_camera_object:D.object_pose,pose_source:D.pose_path,CAD_overlay_source:D.overlay_path},null,2);
 draw('original',img);draw('cad',overlay);
}
$('candidate').onchange=()=>{index=Number($('candidate').value);update()};$('slider').oninput=()=>{index=Number($('slider').value);update()};
$('prev').onclick=()=>{index--;update()};$('next').onclick=()=>{index++;update()};
for(const id of ['zoom','mask','origin','axis','contacts'])$(id).onchange=update;
img.onload=update;overlay.onload=update;img.src=D.rgb;overlay.src=D.overlay;update();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/uon/data1/Dataset_2026"))
    parser.add_argument("--scene", default="Home/Uon_1F_Home/white_table")
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--camera", default="top_view_camera")
    parser.add_argument("--object", default="black_pepper_shaker")
    parser.add_argument("--max-grasps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tiptop-stage1-viewer.html"))
    args = parser.parse_args()
    if args.max_grasps < 1:
        parser.error("--max-grasps must be positive")
    if args.output.suffix.lower() != ".html" or args.output.resolve().is_relative_to(args.dataset_root.resolve()):
        parser.error("Output must be an HTML file outside the input dataset")
    data = build(args)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    print(f"뷰어: {args.output.resolve()}\n후보: {len(data['rows'])}개\n열기: xdg-open {args.output.resolve()}")


if __name__ == "__main__":
    main()

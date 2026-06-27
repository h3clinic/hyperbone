"""
Build 3D prediction viewer: GT armature vs HyperBone3D predicted armature.

Usage:
  python scripts/build_prediction_3d_viewer.py \
    --predictions outputs/models/hyperbone3d_v0_fox_split/eval_val/predictions.jsonl \
    --gt output/fox3d/fox_armature_gt.jsonl \
    --out outputs/models/hyperbone3d_v0_fox_split/eval_val/viewer
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import shutil
from pathlib import Path

from hyperbone.pose3d.joint_map import QUADRUPED_JOINTS, QUADRUPED_BONES, NUM_JOINTS


VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HyperBone3D Prediction Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; height: 100vh; }
#canvas-wrap { flex: 1; position: relative; }
canvas { display: block; width: 100%; height: 100%; }
#panel { width: 340px; background: #16213e; padding: 16px; overflow-y: auto; border-left: 1px solid #333; }
h1 { font-size: 15px; color: #4fc3f7; margin-bottom: 10px; }
h2 { font-size: 12px; color: #81d4fa; margin: 10px 0 4px; }
.row { display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0; border-bottom: 1px solid #222; }
.lbl { color: #aaa; }
.val { color: #fff; font-weight: 600; }
.ctrl { margin: 10px 0; }
.ctrl button { background: #0f3460; color: #ccc; border: 1px solid #4fc3f7; padding: 5px 10px; cursor: pointer; border-radius: 3px; font-size: 11px; margin: 2px; }
.ctrl button:hover { background: #4fc3f7; color: #000; }
input[type=range] { width: 100%; margin: 6px 0; }
.toggle { display: flex; align-items: center; gap: 6px; margin: 3px 0; font-size: 11px; }
.toggle input { accent-color: #4fc3f7; }
.legend { margin-top: 10px; font-size: 11px; }
.leg-item { display: flex; align-items: center; gap: 5px; margin: 2px 0; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
#status { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.7); padding: 6px 10px; border-radius: 4px; font-size: 10px; }
table { width: 100%; font-size: 10px; border-collapse: collapse; margin-top: 6px; }
td { padding: 1px 4px; border-bottom: 1px solid #222; }
td:first-child { color: #aaa; }
td:last-child { text-align: right; color: #fff; }
</style>
</head>
<body>
<div id="canvas-wrap">
  <canvas id="c3d"></canvas>
  <div id="status">Loading...</div>
</div>
<div id="panel">
  <h1>HyperBone3D Prediction Viewer</h1>
  <div class="ctrl">
    <button id="bPlay">▶ Play</button>
    <button id="bPause">⏸</button>
    <button id="bPrev">◀</button>
    <button id="bNext">▶</button>
  </div>
  <div>
    <label style="font-size:11px">Frame: <span id="fLbl">0</span></label>
    <input type="range" id="fSlider" min="0" max="0" value="0">
  </div>
  <h2>Metrics</h2>
  <div class="row"><span class="lbl">MPJPE</span><span class="val" id="mMPJPE">-</span></div>
  <div class="row"><span class="lbl">Vis joints</span><span class="val" id="mVis">-</span></div>
  <div class="row"><span class="lbl">PCK@0.10</span><span class="val" id="mPCK">-</span></div>
  <h2>Toggles</h2>
  <div class="toggle"><input type="checkbox" id="tGT" checked><label for="tGT">GT Armature</label></div>
  <div class="toggle"><input type="checkbox" id="tPred" checked><label for="tPred">Predicted</label></div>
  <div class="toggle"><input type="checkbox" id="tNames" ><label for="tNames">Joint Names</label></div>
  <h2>Legend</h2>
  <div class="legend">
    <div class="leg-item"><div class="dot" style="background:#00ffff"></div>GT Joint</div>
    <div class="leg-item"><div class="dot" style="background:#ff4444"></div>Predicted Joint</div>
    <div class="leg-item"><div style="width:20px;height:2px;background:#ffc800"></div>GT Bone</div>
    <div class="leg-item"><div style="width:20px;height:2px;background:#44ff44"></div>Pred Bone</div>
  </div>
  <h2>Per-Joint Error</h2>
  <table id="jointTable"></table>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const BONES = BONE_DATA_PLACEHOLDER;
const JOINT_NAMES = JOINT_NAMES_PLACEHOLDER;
let gtData = [], predData = [], currentFrame = 0, playing = false;

async function init() {
  const [gtResp, predResp] = await Promise.all([
    fetch('data/gt_frames.json'), fetch('data/pred_frames.json')
  ]);
  gtData = await gtResp.json();
  predData = await predResp.json();
  document.getElementById('fSlider').max = Math.max(gtData.length - 1, 0);
  document.getElementById('status').textContent = `Loaded ${gtData.length} GT + ${predData.length} pred frames`;
  setup3D();
  render();
}

let scene, camera, renderer, controls;
let gtGroup, predGroup;

function setup3D() {
  const canvas = document.getElementById('c3d');
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  camera = new THREE.PerspectiveCamera(50, w/h, 0.1, 1000);
  camera.position.set(3, 2, 3);
  renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  renderer.setSize(w, h);
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  // Grid
  scene.add(new THREE.GridHelper(4, 10, 0x333333, 0x222222));
  // Groups
  gtGroup = new THREE.Group(); scene.add(gtGroup);
  predGroup = new THREE.Group(); scene.add(predGroup);
  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function render() {
  // Clear groups
  while(gtGroup.children.length) gtGroup.remove(gtGroup.children[0]);
  while(predGroup.children.length) predGroup.remove(predGroup.children[0]);

  const showGT = document.getElementById('tGT').checked;
  const showPred = document.getElementById('tPred').checked;

  if (currentFrame < gtData.length && showGT) drawSkeleton(gtData[currentFrame], gtGroup, 0x00ffff, 0xffc800);
  if (currentFrame < predData.length && showPred) drawSkeleton(predData[currentFrame], predGroup, 0xff4444, 0x44ff44);

  // Metrics
  if (currentFrame < predData.length && currentFrame < gtData.length) {
    const gt = gtData[currentFrame], pr = predData[currentFrame];
    let errors = [];
    for (let j = 0; j < Math.min(gt.xyz.length, pr.xyz.length); j++) {
      if (gt.vis[j] > 0.5) {
        const dx = gt.xyz[j][0]-pr.xyz[j][0], dy = gt.xyz[j][1]-pr.xyz[j][1], dz = gt.xyz[j][2]-pr.xyz[j][2];
        errors.push(Math.sqrt(dx*dx+dy*dy+dz*dz));
      }
    }
    const mpjpe = errors.length ? (errors.reduce((a,b)=>a+b)/errors.length).toFixed(4) : '-';
    const pck10 = errors.length ? (errors.filter(e=>e<0.1).length/errors.length*100).toFixed(0)+'%' : '-';
    document.getElementById('mMPJPE').textContent = mpjpe;
    document.getElementById('mVis').textContent = gt.vis.filter(v=>v>0.5).length + '/' + gt.vis.length;
    document.getElementById('mPCK').textContent = pck10;
    // Per-joint table
    let html = '';
    for (let j = 0; j < JOINT_NAMES.length; j++) {
      if (j < errors.length && gt.vis[j] > 0.5) {
        const err = (currentFrame < predData.length) ? errors[j] : null;
        html += `<tr><td>${JOINT_NAMES[j]}</td><td>${err !== null ? err.toFixed(4) : '-'}</td></tr>`;
      }
    }
    document.getElementById('jointTable').innerHTML = html;
  }

  document.getElementById('fLbl').textContent = currentFrame;
  document.getElementById('fSlider').value = currentFrame;
}

function drawSkeleton(frame, group, jointColor, boneColor) {
  const xyz = frame.xyz;
  const jMat = new THREE.MeshBasicMaterial({color: jointColor});
  const bMat = new THREE.LineBasicMaterial({color: boneColor, linewidth: 2});
  for (let j = 0; j < xyz.length; j++) {
    if (frame.vis[j] < 0.5) continue;
    const geo = new THREE.SphereGeometry(0.03, 8, 8);
    const mesh = new THREE.Mesh(geo, jMat);
    mesh.position.set(xyz[j][0], xyz[j][1], xyz[j][2]);
    group.add(mesh);
  }
  for (const [pi, ci] of BONES) {
    if (pi >= xyz.length || ci >= xyz.length) continue;
    if (frame.vis[pi] < 0.5 || frame.vis[ci] < 0.5) continue;
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(xyz[pi][0], xyz[pi][1], xyz[pi][2]),
      new THREE.Vector3(xyz[ci][0], xyz[ci][1], xyz[ci][2]),
    ]);
    group.add(new THREE.Line(geo, bMat));
  }
}

// Controls
document.getElementById('bPlay').onclick = () => { playing = true; playLoop(); };
document.getElementById('bPause').onclick = () => { playing = false; };
document.getElementById('bPrev').onclick = () => { currentFrame = Math.max(0, currentFrame-1); render(); };
document.getElementById('bNext').onclick = () => { currentFrame = Math.min(gtData.length-1, currentFrame+1); render(); };
document.getElementById('fSlider').oninput = (e) => { currentFrame = +e.target.value; render(); };
document.getElementById('tGT').onchange = render;
document.getElementById('tPred').onchange = render;

function playLoop() {
  if (!playing) return;
  currentFrame = (currentFrame + 1) % gtData.length;
  render();
  setTimeout(playLoop, 42); // ~24fps
}

init();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build HyperBone3D prediction 3D viewer")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL")
    parser.add_argument("--gt", required=True, help="GT armature JSONL")
    parser.add_argument("--out", required=True, help="Output viewer directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Load predictions
    pred_records = []
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                pred_records.append(json.loads(line))
    print(f"Loaded {len(pred_records)} predictions")

    # Load GT
    gt_records = []
    with open(args.gt) as f:
        for line in f:
            if line.strip():
                gt_records.append(json.loads(line))
    print(f"Loaded {len(gt_records)} GT records")

    # Convert to viewer format
    # Predictions have pred_xyz [J,3] and pred_vis [J] in canonical space
    pred_frames = []
    for r in pred_records:
        pred_frames.append({
            "frame_idx": r["frame_idx"],
            "xyz": r["pred_xyz"],
            "vis": r["pred_vis"],
        })

    # GT: remap to canonical space for matching
    from hyperbone.pose3d.joint_map import remap_joints_to_canonical
    import numpy as np

    gt_frames = []
    for r in gt_records:
        # Only include GT frames that have matching predictions
        pred_frame_idxs = {p["frame_idx"] for p in pred_records}
        if r["frame_idx"] not in pred_frame_idxs:
            continue

        joints = r.get("joints", [])
        xyz_list, vis_list = remap_joints_to_canonical(joints, "Fox.glb")
        xyz_arr = np.array(xyz_list, dtype=np.float32)
        root_xyz = xyz_arr[0]
        scale = np.linalg.norm(xyz_arr[3] - xyz_arr[0]) if vis_list[3] else 1.0
        scale = max(scale, 1e-3)
        canonical = ((xyz_arr - root_xyz) / scale).tolist()

        gt_frames.append({
            "frame_idx": r["frame_idx"],
            "xyz": canonical,
            "vis": [1.0 if v else 0.0 for v in vis_list],
        })

    # Sort by frame
    gt_frames.sort(key=lambda x: x["frame_idx"])
    pred_frames.sort(key=lambda x: x["frame_idx"])

    # Write data files
    with open(data_dir / "gt_frames.json", 'w') as f:
        json.dump(gt_frames, f)
    with open(data_dir / "pred_frames.json", 'w') as f:
        json.dump(pred_frames, f)

    # Write HTML (replace placeholders)
    html = VIEWER_HTML.replace(
        "BONE_DATA_PLACEHOLDER",
        json.dumps(QUADRUPED_BONES)
    ).replace(
        "JOINT_NAMES_PLACEHOLDER",
        json.dumps(QUADRUPED_JOINTS)
    )

    with open(out_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nViewer built: {out_dir}")
    print(f"  GT frames: {len(gt_frames)}")
    print(f"  Pred frames: {len(pred_frames)}")
    print(f"  Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

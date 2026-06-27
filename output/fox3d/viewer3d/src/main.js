/**
 * HyperBone Pose3D Viewer - Main entry point.
 * 
 * Loads GT armature and HyperBone canonical graph, displays both in 3D.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadDataManifest, loadGTFrames, loadHyperBoneFrames } from './loaders.js';
import { createGTSkeleton, createHyperBoneSkeleton, createBboxPlane } from './scene.js';

// ─── State ──────────────────────────────────────────────────────────────────
let gtFrames = [];
let hbFrames = [];
let currentFrame = 0;
let playing = false;
let animId = null;

let scene, camera, renderer, controls;
let gtGroup = null, hbGroup = null, bboxGroup = null;

// ─── Init Three.js ──────────────────────────────────────────────────────────
function initScene() {
    const canvas = document.getElementById('canvas3d');
    const container = document.getElementById('viewer3d');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.01, 100);
    camera.position.set(1.5, 1.0, 1.5);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.target.set(0, 0, 0);

    // Grid
    const grid = new THREE.GridHelper(2, 20, 0x333333, 0x222222);
    scene.add(grid);

    // Axes
    const axes = new THREE.AxesHelper(0.5);
    scene.add(axes);

    // Ambient light
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    animate();
}

function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// ─── Frame Management ───────────────────────────────────────────────────────
function showFrame(idx) {
    currentFrame = idx;

    // Remove old groups
    if (gtGroup) { scene.remove(gtGroup); }
    if (hbGroup) { scene.remove(hbGroup); }
    if (bboxGroup) { scene.remove(bboxGroup); }

    const showGT = document.getElementById('togGT').checked;
    const showHB = document.getElementById('togHB').checked;
    const showBbox = document.getElementById('togBbox').checked;

    // GT frame
    const gtFrame = gtFrames[idx] || null;
    if (gtFrame) {
        gtGroup = createGTSkeleton(gtFrame, scene, showGT);
    }

    // HyperBone frame (match by frame_idx)
    const hbFrame = hbFrames.find(f => f.frame_idx === (gtFrame ? gtFrame.frame_idx : idx)) || hbFrames[idx] || null;
    if (hbFrame) {
        hbGroup = createHyperBoneSkeleton(hbFrame, scene, showHB);
    }

    // Bbox
    if (showBbox && (gtFrame || hbFrame)) {
        const bbox = (hbFrame && hbFrame.bbox_xywh) || (gtFrame && gtFrame.bbox_xywh);
        const res = (gtFrame && gtFrame.resolution) || [640, 480];
        if (bbox) {
            bboxGroup = createBboxPlane(bbox, res, scene, true);
        }
    }

    // Update UI
    updateMetrics(gtFrame, hbFrame);
    document.getElementById('frameLabel').textContent = idx;
    document.getElementById('frameSlider').value = idx;
}

function updateMetrics(gtFrame, hbFrame) {
    document.getElementById('metFrame').textContent = currentFrame;
    document.getElementById('metGTJoints').textContent = gtFrame ? gtFrame.joints.length : '-';
    document.getElementById('metGTBones').textContent = gtFrame ? (gtFrame.bones ? gtFrame.bones.length : '-') : '-';
    document.getElementById('metHBNodes').textContent = hbFrame ? (hbFrame.nodes ? hbFrame.nodes.length : '-') : '-';
    document.getElementById('metHBEdges').textContent = hbFrame ? (hbFrame.edges ? hbFrame.edges.length : '-') : '-';
    document.getElementById('metAccepted').textContent = hbFrame ? (hbFrame.accepted ? '✓' : '✗') : '-';

    // NN error and coverage (compute if both exist)
    if (gtFrame && hbFrame && gtFrame.joints && hbFrame.nodes) {
        const { nnError, coverage } = computeNNMetrics(gtFrame, hbFrame);
        document.getElementById('metNNError').textContent = nnError.toFixed(1);
        document.getElementById('metCoverage').textContent = (coverage * 100).toFixed(0) + '%';
    } else {
        document.getElementById('metNNError').textContent = '-';
        document.getElementById('metCoverage').textContent = '-';
    }
}

function computeNNMetrics(gtFrame, hbFrame) {
    // For each GT joint projected to image, find nearest HB node
    const gtPts = gtFrame.joints.filter(j => j.visible).map(j => j.image_xy);
    const hbPts = hbFrame.nodes.map(n => n.image_xy);

    if (!gtPts.length || !hbPts.length) return { nnError: Infinity, coverage: 0 };

    let totalDist = 0;
    let covered = 0;
    const threshold = 40; // pixels

    gtPts.forEach(gp => {
        let minDist = Infinity;
        hbPts.forEach(hp => {
            const dx = gp[0] - hp[0];
            const dy = gp[1] - hp[1];
            const d = Math.sqrt(dx*dx + dy*dy);
            if (d < minDist) minDist = d;
        });
        totalDist += minDist;
        if (minDist < threshold) covered++;
    });

    return {
        nnError: totalDist / gtPts.length,
        coverage: covered / gtPts.length,
    };
}

// ─── Playback ───────────────────────────────────────────────────────────────
let playInterval = null;

function play() {
    if (playInterval) return;
    playing = true;
    const maxFrame = Math.max(gtFrames.length, hbFrames.length) - 1;
    playInterval = setInterval(() => {
        currentFrame = (currentFrame + 1) % (maxFrame + 1);
        showFrame(currentFrame);
    }, 1000 / 24);
}

function pause() {
    playing = false;
    if (playInterval) { clearInterval(playInterval); playInterval = null; }
}

// ─── Load Data & Init ───────────────────────────────────────────────────────
async function init() {
    initScene();

    const status = document.getElementById('status');

    try {
        const manifest = await loadDataManifest('.');
        status.textContent = 'Loading data...';

        if (manifest.gt_path) {
            gtFrames = await loadGTFrames(manifest.gt_path);
        }
        if (manifest.hyperbone_path) {
            hbFrames = await loadHyperBoneFrames(manifest.hyperbone_path);
        }

        const maxFrame = Math.max(gtFrames.length, hbFrames.length) - 1;
        document.getElementById('frameSlider').max = maxFrame;

        status.textContent = `GT: ${gtFrames.length} frames | HB: ${hbFrames.length} frames`;
        showFrame(0);

    } catch (e) {
        status.textContent = `Error: ${e.message}. Place data_manifest.json alongside index.html.`;
        console.error(e);
    }

    // Wire up controls
    document.getElementById('btnPlay').addEventListener('click', play);
    document.getElementById('btnPause').addEventListener('click', pause);
    document.getElementById('btnPrev').addEventListener('click', () => { pause(); showFrame(Math.max(0, currentFrame - 1)); });
    document.getElementById('btnNext').addEventListener('click', () => { pause(); showFrame(currentFrame + 1); });
    document.getElementById('frameSlider').addEventListener('input', (e) => { pause(); showFrame(parseInt(e.target.value)); });

    // Toggles
    ['togGT', 'togHB', 'togBbox', 'togProjected', 'togJointNames'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => showFrame(currentFrame));
    });
}

init();

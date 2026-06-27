/**
 * Three.js scene management for pose3d viewer.
 */
import * as THREE from 'three';

export function createGTSkeleton(frameData, scene, visible) {
    const group = new THREE.Group();
    group.name = 'gt_skeleton';
    group.visible = visible;

    if (!frameData || !frameData.joints) return group;

    const joints = frameData.joints;
    const jointMap = {};

    // Draw joints as spheres
    const jointGeo = new THREE.SphereGeometry(0.02, 8, 8);
    const jointMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });

    joints.forEach(j => {
        const mesh = new THREE.Mesh(jointGeo, jointMat);
        const pos = j.world_xyz || j.canonical_xyz || [0, 0, 0];
        mesh.position.set(pos[0], pos[2] || 0, -pos[1]);  // Convert coords
        mesh.userData = { name: j.name, id: j.id };
        group.add(mesh);
        jointMap[j.id] = mesh.position.clone();
    });

    // Draw bones as lines
    const boneMat = new THREE.LineBasicMaterial({ color: 0x0097a7, linewidth: 2 });
    joints.forEach(j => {
        if (j.parent_id !== null && j.parent_id !== undefined && jointMap[j.parent_id]) {
            const points = [jointMap[j.parent_id], jointMap[j.id]];
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geo, boneMat);
            group.add(line);
        }
    });

    scene.add(group);
    return group;
}

export function createHyperBoneSkeleton(frameData, scene, visible) {
    const group = new THREE.Group();
    group.name = 'hyperbone_skeleton';
    group.visible = visible;

    if (!frameData || !frameData.nodes) return group;

    const nodes = frameData.nodes;
    const nodeMap = {};

    // Draw nodes
    const nodeGeo = new THREE.SphereGeometry(0.015, 6, 6);
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0xffeb3b });

    nodes.forEach(n => {
        const mesh = new THREE.Mesh(nodeGeo, nodeMat);
        const pos = n.canonical_xyz || [0, 0, 0];
        mesh.position.set(pos[0], pos[2] || 0, -pos[1]);
        group.add(mesh);
        nodeMap[n.id] = mesh.position.clone();
    });

    // Draw edges
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xff9800, linewidth: 1 });
    const edges = frameData.edges || [];
    edges.forEach(e => {
        const src = nodeMap[e.source];
        const tgt = nodeMap[e.target];
        if (src && tgt) {
            const geo = new THREE.BufferGeometry().setFromPoints([src, tgt]);
            const line = new THREE.Line(geo, edgeMat);
            group.add(line);
        }
    });

    scene.add(group);
    return group;
}

export function createBboxPlane(bbox_xywh, resolution, scene, visible) {
    const group = new THREE.Group();
    group.name = 'bbox_plane';
    group.visible = visible;

    if (!bbox_xywh) return group;

    const [bx, by, bw, bh] = bbox_xywh;
    const [resW, resH] = resolution || [640, 480];

    // Normalize to [-0.5, 0.5]
    const x = (bx / resW) - 0.5;
    const y = 0.5 - (by / resH);
    const w = bw / resW;
    const h = bh / resH;

    const geo = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({
        color: 0x00ff00, wireframe: true, transparent: true, opacity: 0.3
    });
    const plane = new THREE.Mesh(geo, mat);
    plane.position.set(x + w/2, y - h/2, -0.3);
    group.add(plane);

    scene.add(group);
    return group;
}

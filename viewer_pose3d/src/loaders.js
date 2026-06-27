/**
 * Load pose3d data from JSONL files.
 */

export async function loadJsonl(url) {
    const resp = await fetch(url);
    const text = await resp.text();
    const lines = text.split('\n').filter(l => l.trim());
    return lines.map(l => JSON.parse(l));
}

export async function loadDataManifest(baseUrl) {
    const resp = await fetch(`${baseUrl}/data_manifest.json`);
    return resp.json();
}

export async function loadGTFrames(url) {
    const frames = await loadJsonl(url);
    console.log(`[Loader] Loaded ${frames.length} GT frames`);
    return frames;
}

export async function loadHyperBoneFrames(url) {
    const frames = await loadJsonl(url);
    console.log(`[Loader] Loaded ${frames.length} HyperBone frames`);
    return frames;
}

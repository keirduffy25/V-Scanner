// V-Scanner V1 – robust startup with on-screen diagnostics for iPad/GitHub Pages
// Assumes index.html loads yolo.min.js BEFORE this file and has <canvas id="hud"></canvas>

////////// UI: tiny HUD logger so you can see what's wrong on iPad //////////
const hud = (() => {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed;left:8px;bottom:8px;max-width:90vw;z-index:9999;
    background:#000c;color:#0ff;border:1px solid #044;box-shadow:0 0 10px #0ff;
    padding:8px 10px;font:12px ui-monospace,monospace;line-height:1.45;white-space:pre-wrap`;
  el.textContent = 'INIT…';
  addEventListener('DOMContentLoaded', () => document.body.appendChild(el));
  return {
    set: (t) => (el.textContent = t),
    add: (t) => (el.textContent += `\n${t}`),
  };
})();

function inIframe() {
  try { return window.self !== window.top; } catch (_) { return true; }
}

////////// MODEL URLS: absolute path for GitHub Pages + mirrors //////////
const LOCAL_MODEL = location.hostname.includes('github.io')
  ? '/V-Scanner/yolov8n.onnx'       // served by your Pages site
  : './yolov8n.onnx';               // local dev

const MODEL_URLS = [
  LOCAL_MODEL,
  'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
  'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx',
];

////////// Quick network probe so you see WHY a model URL fails //////////
async function probe(url) {
  try {
    // Use GET with small range to avoid big downloads; HEAD is flaky on some CDNs
    const res = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
    return { ok: res.ok, status: res.status, url: res.url };
  } catch (e) {
    return { ok: false, status: 'network-error', url, err: e.message || String(e) };
  }
}

////////// Camera startup with friendly errors //////////
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Camera API not available (WebKit permissions or browser feature missing).');
  }
  // iOS needs playsinline/autoplay muted
  const v = document.createElement('video');
  v.setAttribute('playsinline', '');
  v.setAttribute('autoplay', '');
  v.muted = true;
  v.style.display = 'none';
  document.body.appendChild(v);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    v.srcObject = stream;
    await new Promise((r) => (v.readyState >= 2 ? r() : (v.onloadedmetadata = r)));
    return v;
  } catch (e) {
    // Most common: permission denied on iPad
    const hint =
      'Camera blocked.\nSafari: AA ▸ Website Settings ▸ Camera ▸ Allow.\nChrome: ⋮ ▸ Site Settings ▸ Camera ▸ Allow.\nThen reload.';
    throw new Error(`${e.name || 'getUserMedia'}: ${e.message || e}. ${hint}`);
  }
}

////////// YOLO loader with fallbacks + inline diagnostics //////////
async function loadModel() {
  // Make sure yolo.min.js is actually present
  if (!window.YOLO || !YOLO.load) {
    throw new Error('yolo.min.js not loaded. Ensure the script tag is BEFORE script.js in index.html.');
  }

  // Probe each URL first so we can print a helpful reason
  for (const url of MODEL_URLS) {
    const p = await probe(url);
    hud.add(`Probe: ${p.url} → ${p.ok ? '200 OK' : p.status}`);
    if (!p.ok) continue;
    try {
      hud.add(`Loading model from: ${p.url}`);
      const m = await YOLO.load(p.url);
      hud.add('✅ Model loaded');
      return m;
    } catch (e) {
      hud.add(`❌ YOLO.load() failed from ${p.url}: ${e.message || e}`);
    }
  }

  throw new Error(
    'Failed to load YOLO model from all sources.\n' +
    `Check: ${location.origin}/V-Scanner/yolov8n.onnx opens/downloads.\n` +
    'If 404, upload yolov8n.onnx next to index.html (repo root).'
  );
}

////////// Drawing helpers //////////
function fitCanvasToVideo(canvas, video) {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}
function glowStrokeRect(ctx, x, y, w, h, color = '#00FFFF') {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}
function label(ctx, text, x, y, color = '#00FFFF') {
  ctx.save();
  ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  ctx.fillText(text, x, y);
  ctx.restore();
}

////////// MAIN /////////
(async () => {
  hud.set('V-Scanner starting…');

  if (inIframe()) {
    hud.add('⚠️ Page is in an embedded preview/iframe. Open the page directly:\n' +
            'https://keirduffy25.github.io/V-Scanner/');
  }

  try {
    hud.add('Starting camera…');
    const video = await startCamera();
    hud.add('✅ Camera ready');

    const canvas = document.getElementById('hud');
    const ctx = canvas.getContext('2d', { alpha: true });
    fitCanvasToVideo(canvas, video);
    addEventListener('resize', () => fitCanvasToVideo(canvas, video));

    hud.add('Loading YOLOv8n…');
    const model = await loadModel();

    hud.add('🚀 Scanner running');
    async function loop() {
      // background video
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      let preds = [];
      try {
        preds = await model.detect(video); // [{box:[x,y,w,h], class, score}]
      } catch (e) {
        // Print once per N frames if needed
      }

      if (preds && preds.length) {
        const best = preds.reduce((a, b) => (a.score > b.score ? a : b));
        const [x, y, w, h] = best.box;
        const text = `${best.class.toUpperCase()} — ${Math.round(best.score * 100)}%`;
        glowStrokeRect(ctx, x, y, w, h);
        label(ctx, text, x + w / 2, Math.max(16, y - 8));
      } else {
        label(ctx, 'SCANNING…', canvas.width / 2, canvas.height / 2);
      }

      requestAnimationFrame(loop);
    }
    loop();
  } catch (err) {
    console.error(err);
    hud.add(`\n❌ Startup error:\n${err.message || err}`);
    alert(err.message || String(err));
  }
})();

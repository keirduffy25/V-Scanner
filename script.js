// Cyberpunk V-Scanner (Stable Version)
// Includes absolute model URL, camera fixes, and diagnostics for iPad + GitHub Pages

// Small on-screen diagnostic log (visible at bottom-left)
const hud = (() => {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed;left:8px;bottom:8px;max-width:90vw;z-index:9999;
    background:#000c;color:#0ff;border:1px solid #044;box-shadow:0 0 10px #0ff;
    padding:8px 10px;font:12px ui-monospace,monospace;line-height:1.45;white-space:pre-wrap`;
  el.textContent = 'Starting V-Scanner...';
  addEventListener('DOMContentLoaded', () => document.body.appendChild(el));
  return {
    set: (t) => (el.textContent = t),
    add: (t) => (el.textContent += `\n${t}`)
  };
})();

function inIframe() {
  try { return window.self !== window.top; } catch (_) { return true; }
}

// ✅ Hard-coded model URLs (absolute + mirrors)
const MODEL_URLS = [
  'https://keirduffy25.github.io/V-Scanner/yolov8n.onnx', // your hosted model
  'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx', // fallback
  'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx' // backup
];

// Probe URLs to show which work
async function probe(url) {
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
    return { ok: res.ok, status: res.status, url: res.url };
  } catch (e) {
    return { ok: false, status: 'network-error', url, err: e.message || String(e) };
  }
}

// Start camera with permission handling
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Camera API not available. Use Safari/Chrome with camera access.');
  }
  const video = document.createElement('video');
  video.setAttribute('playsinline', '');
  video.setAttribute('autoplay', '');
  video.muted = true;
  video.style.display = 'none';
  document.body.appendChild(video);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = stream;
    await new Promise((r) => (video.readyState >= 2 ? r() : (video.onloadedmetadata = r)));
    return video;
  } catch (e) {
    const hint =
      'Camera blocked.\nSafari: AA ▸ Website Settings ▸ Camera ▸ Allow.\nChrome: ⋮ ▸ Site Settings ▸ Camera ▸ Allow.';
    throw new Error(`${e.name || 'getUserMedia'}: ${e.message || e}. ${hint}`);
  }
}

// Load YOLO model from any working URL
async function loadModel() {
  if (!window.YOLO || !YOLO.load) {
    throw new Error('yolo.min.js not loaded before script.js');
  }

  for (const url of MODEL_URLS) {
    const p = await probe(url);
    hud.add(`Probe: ${p.url} → ${p.ok ? '200 OK' : p.status}`);
    if (!p.ok) continue;

    try {
      hud.add(`Loading model from ${p.url}`);
      const m = await YOLO.load(p.url);
      hud.add('✅ YOLO model loaded successfully');
      return m;
    } catch (e) {
      hud.add(`❌ Failed from ${p.url}: ${e.message || e}`);
    }
  }

  throw new Error(
    'Failed to load YOLO model. Ensure yolov8n.onnx is accessible.\n' +
    'Try opening it directly: https://keirduffy25.github.io/V-Scanner/yolov8n.onnx'
  );
}

// Drawing helpers
function fitCanvasToVideo(canvas, video) {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}
function glowRect(ctx, x, y, w, h, color = '#0ff') {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}
function drawLabel(ctx, text, x, y, color = '#0ff') {
  ctx.save();
  ctx.font = '14px ui-monospace,monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  ctx.fillText(text, x, y);
  ctx.restore();
}

// Main
(async () => {
  hud.set('V-Scanner initializing...');

  if (inIframe()) {
    hud.add('⚠️ You are in an embedded view. Open directly:\nhttps://keirduffy25.github.io/V-Scanner/');
  }

  try {
    hud.add('Starting camera...');
    const video = await startCamera();
    hud.add('✅ Camera ready');

    const canvas = document.getElementById('hud');
    const ctx = canvas.getContext('2d', { alpha: true });
    fitCanvasToVideo(canvas, video);
    addEventListener('resize', () => fitCanvasToVideo(canvas, video));

    hud.add('Loading YOLO model...');
    const model = await loadModel();

    hud.add('🚀 Scanner running');
    async function loop() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      let preds = [];
      try {
        preds = await model.detect(video);
      } catch (e) {
        // Ignore transient model errors
      }

      if (preds && preds.length) {
        const best = preds.reduce((a, b) => (a.score > b.score ? a : b));
        const [x, y, w, h] = best.box;
        glowRect(ctx, x, y, w, h);
        drawLabel(ctx, `${best.class} ${Math.round(best.score * 100)}%`, x + w / 2, y - 5);
      } else {
        drawLabel(ctx, 'Scanning...', canvas.width / 2, canvas.height / 2);
      }

      requestAnimationFrame(loop);
    }
    loop();
  } catch (err) {
    console.error(err);
    hud.add(`❌ Error: ${err.message || err}`);
    alert(err.message || String(err));
  }
})();

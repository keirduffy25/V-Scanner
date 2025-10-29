// Cyberpunk V-Scanner — Version 1 (YOLOv8n, GitHub Pages path fix)
// Requires: index.html with <canvas id="hud"> and yolo.min.js loaded before this file.

// ---------- tiny on-screen status (handy on iPad) ----------
const statusEl = (() => {
  const el = document.createElement('div');
  el.style.cssText =
    'position:fixed;left:8px;bottom:8px;padding:6px 8px;background:#000a;color:#0ff;font:12px monospace;border:1px solid #044;box-shadow:0 0 8px #0ff;z-index:9';
  el.textContent = 'INIT…';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(el));
  return el;
})();
const setStatus = (t) => (statusEl.textContent = t);

// ---------- camera ----------
async function startCamera() {
  setStatus('Requesting camera…');
  const v = document.createElement('video');
  v.setAttribute('playsinline', '');
  v.setAttribute('autoplay', '');
  v.muted = true;
  v.style.display = 'none';
  document.body.appendChild(v);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  v.srcObject = stream;
  await new Promise((r) => (v.readyState >= 2 ? r() : (v.onloadedmetadata = r)));
  setStatus('✅ Camera ready');
  return v;
}

// ---------- YOLO loader with GitHub Pages path fix & fallbacks ----------
async function loadModelWithFallback() {
  setStatus('Loading YOLOv8n model…');

  // If site runs on *.github.io, use an absolute path to your repo folder.
  // Otherwise (local dev), use a relative path.
  const LOCAL_MODEL = location.hostname.includes('github.io')
    ? '/V-Scanner/yolov8n.onnx'
    : './yolov8n.onnx';

  const urls = [
    LOCAL_MODEL, // your local copy in the repo root
    'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
    'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx'
  ];

  let lastErr = null;
  for (const url of urls) {
    try {
      setStatus(`Loading model from: ${url}`);
      const model = await YOLO.load(url); // provided by yolo.min.js
      setStatus('✅ Model loaded');
      return model;
    } catch (e) {
      lastErr = e;
      console.warn('[YOLO] failed from', url, e);
    }
  }

  setStatus('❌ Failed to load model');
  alert('Failed to load YOLO model.\nEnsure yolov8n.onnx sits next to index.html (repo root).');
  throw lastErr ?? new Error('Model load failed (all sources).');
}

// ---------- draw helpers ----------
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
function labelText(ctx, text, x, y, color = '#00FFFF') {
  ctx.save();
  ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  ctx.fillText(text, x, y);
  ctx.restore();
}

// ---------- main ----------
(async function main() {
  try {
    const video = await startCamera();

    const canvas = document.getElementById('hud');
    const ctx = canvas.getContext('2d', { alpha: true });
    fitCanvasToVideo(canvas, video);
    addEventListener('resize', () => fitCanvasToVideo(canvas, video));

    const model = await loadModelWithFallback();

    setStatus('🚀 Scanner running… (allow camera if prompted)');

    // render loop
    async function loop() {
      // Draw camera frame as background (optional; comment out if you only want HUD)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Run detection
      let preds = [];
      try {
        preds = await model.detect(video); // [{box:[x,y,w,h], class, score}]
      } catch (e) {
        console.warn('detect() error:', e);
      }

      // Show only the single highest-confidence detection (V1 behavior)
      if (preds && preds.length) {
        const best = preds.reduce((a, b) => (a.score > b.score ? a : b));
        const [x, y, w, h] = best.box;
        const label = `${best.class.toUpperCase()} — ${Math.round(best.score * 100)}%`;
        glowStrokeRect(ctx, x, y, w, h, '#00FFFF');
        labelText(ctx, label, x + w / 2, Math.max(16, y - 8), '#00FFFF');
      } else {
        // idle hint
        labelText(ctx, 'SCANNING…', canvas.width / 2, canvas.height / 2, '#00FFFF');
      }

      requestAnimationFrame(loop);
    }
    loop();
  } catch (err) {
    console.error('Fatal startup error:', err);
    setStatus('❌ Startup failed (see console)');
    alert('Could not start camera or model. If viewing inside an embedded preview, open the page directly.');
  }
})();

/* ---------------------------------------------------------
   Cyberpunk V-Scanner — Version 1 (YOLO guard enabled)
   This file assumes:
   - index.html loads yolo.min.js BEFORE this file (or our guard waits)
   - yolov8n.onnx sits next to index.html in the repo root
--------------------------------------------------------- */

// === 0) Small HUD logger so we always see what’s happening ===
const hud = (() => {
  let el;
  const ensure = () => {
    if (!el) {
      el = document.createElement('div');
      el.style.cssText = `
        position:fixed;left:12px;top:12px;z-index:9999;
        min-width:200px;max-width:40vw;white-space:pre-line;
        background:#001016cc;border:1px solid #00ffff66;
        border-radius:6px;box-shadow:0 0 10px #00ffff66;
        color:#bff;padding:8px 10px;font:12px ui-monospace,monospace;
      `;
      document.body.appendChild(el);
      log("V-Scanner: initialising…");
    }
  };
  const log = (msg) => { ensure(); el.textContent += (el.textContent ? "\n" : "") + msg; };
  const ok  = (msg) => log("✅ " + msg);
  const warn= (msg) => log("⚠️ " + msg);
  const err = (msg) => log("❌ " + msg);
  return { log, ok, warn, err };
})();

// === 1) Ensure YOLO runtime is truly ready before we touch it ===
async function waitForYoloRuntime(maxWaitMs = 10000) {
  const started = Date.now();
  if (window.YOLO && window.YOLO.load) {
    hud.ok("YOLO runtime already present.");
    return true;
  }
  hud.warn("Waiting for YOLO runtime…");
  while (!(window.YOLO && window.YOLO.load) && (Date.now() - started) < maxWaitMs) {
    await new Promise(r => setTimeout(r, 300));
  }
  if (window.YOLO && window.YOLO.load) {
    hud.ok("YOLO runtime confirmed.");
    return true;
  }
  hud.err("yolo.min.js not loaded before script.js");
  alert("yolo.min.js is not loaded before script.js");
  return false;
}

// === 2) Camera setup (rear camera if available) ===
async function startCamera() {
  const constraints = {
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1280 }, height: { ideal: 720 }
    }
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const video = document.createElement('video');
  video.setAttribute('playsinline', '');
  video.autoplay = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
  hud.ok("Camera ready.");

  return video;
}

// === 3) Canvas overlay for boxes/HUD ===
function setupCanvas() {
  const c = document.getElementById('hud') || (() => {
    const el = document.createElement('canvas');
    el.id = 'hud';
    document.body.appendChild(el);
    return el;
  })();
  const ctx = c.getContext('2d');
  const resize = () => { c.width = window.innerWidth; c.height = window.innerHeight; };
  resize(); window.addEventListener('resize', resize);
  return { c, ctx };
}

// === 4) Load YOLOv8 model with local-first, absolute fallback ===
async function loadYoloModel() {
  // local file beside index.html (GitHub Pages root)
  const LOCAL = 'yolov8n.onnx';
  // absolute fallback (your repo’s Pages URL)
  const ABS = 'https://keirduffy25.github.io/V-Scanner/yolov8n.onnx';

  const sources = [LOCAL, ABS];
  let lastErr;

  for (const url of sources) {
    try {
      hud.log(`Loading YOLO model…\n  → ${url}`);
      const model = await YOLO.load(url); // provided by yolo.min.js
      hud.ok("YOLO model loaded.");
      return model;
    } catch (e) {
      lastErr = e;
      hud.warn(`Model load failed from ${url}`);
    }
  }
  hud.err("All model sources failed.");
  console.error(lastErr);
  throw lastErr;
}

// === 5) Main detection loop (single target, highest confidence) ===
function drawBox(ctx, box, label, score, color = 'rgba(0,255,255,0.9)') {
  const [x, y, w, h] = box; // assume xywh from runtime; adapt if needed
  ctx.save();
  ctx.shadowBlur = 12;
  ctx.shadowColor = color;
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, h);

  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x, y - 18, ctx.measureText(label).width + 60, 18);
  ctx.fillStyle = '#0ff';
  ctx.font = '12px ui-monospace,monospace';
  ctx.fillText(`${label} ${(score*100|0)}%`, x + 6, y - 5);
  ctx.restore();
}

async function run(model, video, ctx) {
  hud.ok("Scanner running.");
  const loop = async () => {
    try {
      // draw video frame as background
      ctx.drawImage(video, 0, 0, ctx.canvas.width, ctx.canvas.height);

      // infer (API varies by lib; this follows vladmandic/yolo)
      // The runtime accepts an HTMLCanvas/HTMLVideo/HTMLImage
      const result = await model.detect(ctx.canvas, { maxResults: 10 });

      // pick the highest confidence detection and draw only that one
      if (result && result.length) {
        const best = result.slice().sort((a,b) => (b.score||0) - (a.score||0))[0];
        // best.box expected as [x,y,w,h] in canvas space; remap if result is normalized
        if (best && best.box) {
          drawBox(ctx, best.box, best.class || best.label || 'object', best.score ?? 0.0);
        }
      }
    } catch (e) {
      // keep the loop alive, but show a one-time HUD error
      console.debug(e);
    }
    requestAnimationFrame(loop);
  };
  loop();
}

// === 6) Boot ===
(async function start() {
  try {
    if (!('mediaDevices' in navigator) || !navigator.mediaDevices.getUserMedia) {
      hud.err("getUserMedia not supported.");
      document.getElementById('nosupport')?.style && (document.getElementById('nosupport').style.display = 'block');
      return;
    }

    const ok = await waitForYoloRuntime();
    if (!ok) return; // we already alerted in the guard

    const video = await startCamera();
    const { ctx } = setupCanvas();

    const model = await loadYoloModel();
    await run(model, video, ctx);

  } catch (e) {
    hud.err(e && e.message ? e.message : String(e));
  }
})();

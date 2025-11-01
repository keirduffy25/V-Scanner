// ===== V-Scanner (no yolo.min.js; uses ONNX Runtime Web only) =====

// ---- tiny HUD helper -------------------------------------------------
const hud = (() => {
  const el = document.getElementById('hud');
  const lines = [];
  function push(msg, cls='') {
    lines.unshift({ msg, cls });
    if (lines.length > 10) lines.pop();
    el.innerHTML = lines.map(l => `<div class="${l.cls||''}">• ${l.msg}</div>`).join('');
  }
  return { push };
})();

// ---- ensure ORT loaded BEFORE we do anything --------------------------
document.addEventListener('DOMContentLoaded', async () => {
  try {
    // Safety: if CDN didn’t load or order broke, stop early with a clear hint
    if (!window.ort || !ort.InferenceSession) {
      alert('onnxruntime-web failed to load before script.js.\nCheck index.html: ORT <script> must appear before script.js and both must use defer.');
      hud.push('Error: onnxruntime-web not loaded before script.js', 'err');
      return;
    }

    hud.push('V-Scanner initialising…');

    // ---- 1) Start camera (rear camera on phones/tablets when available) ----
    const video = document.getElementById('cam');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');

    function resize() {
      const w = video.videoWidth || window.innerWidth;
      const h = video.videoHeight || window.innerHeight;
      canvas.width = w; canvas.height = h;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false
      });
      video.srcObject = stream;

      await new Promise(res => video.onloadedmetadata = res);
      video.play().catch(()=>{});
      resize();
      window.addEventListener('resize', resize);
      hud.push('Camera ready ✓');
    } catch (e) {
      hud.push('Camera permission failed (tap address bar → allow camera).', 'warn');
      console.warn(e);
    }

    // ---- 2) Load YOLOv8 ONNX model (file must sit next to index.html) ----
    const MODEL_URL = './yolov8n.onnx'; // keep the name exactly as uploaded

    // quick existence check (HEAD)
    try {
      const head = await fetch(MODEL_URL, { method: 'HEAD', cache: 'no-store' });
      if (!head.ok) throw new Error(`HTTP ${head.status}`);
    } catch (e) {
      alert('Failed to load YOLO model.\nMake sure yolov8n.onnx is in the repo root, next to index.html.');
      hud.push('Error: yolov8n.onnx not reachable', 'err');
      return;
    }

    hud.push('Loading YOLO model…');

    const session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],   // best cross-device compatibility
      graphOptimizationLevel: 'all'
    });

    hud.push('Model loaded ✓');

    // ---- 3) Minimal detection loop (placeholder boxes until you add pre/post) ----
    // NOTE: This is a scaffolding loop to prove run-order & loading are fixed.
    // Hook up your real preprocess/postprocess here.
    const drawDemo = () => {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0,0,w,h);
      // cyberpunk scan glow box (demo)
      const t = Date.now()/500;
      const bw = Math.round(w*0.45 + Math.sin(t)*20);
      const bh = Math.round(h*0.36 + Math.cos(t)*16);
      const x = Math.round((w - bw)/2);
      const y = Math.round((h - bh)/2);

      ctx.shadowColor = '#0ff';
      ctx.shadowBlur = 18;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#70ffff';
      ctx.strokeRect(x, y, bw, bh);

      ctx.shadowBlur = 0;
      ctx.fillStyle = '#bff';
      ctx.font = '12px ui-monospace, Menlo, Consolas, monospace';
      ctx.fillText('V-SCAN ACTIVE (demo box)', x+8, y-8);
      requestAnimationFrame(drawDemo);
    };
    drawDemo();

    // If you want to test a real inference call (once you wire preprocess):
    // const output = await session.run({ images: preprocessedTensor });
    // postprocess(output);

  } catch (err) {
    console.error(err);
    hud.push(`Fatal: ${err.message || err}`, 'err');
  }
});

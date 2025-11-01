/* V-Scanner v1 — YOLOv8 (ONNX Runtime Web) */
// Assumes index.html already includes:
//  - onnxruntime-web (ort) via CDN
//  - yolo.min.js (the small helper used for NMS/decoding)
//  - this script after the above two

// ---- config -------------------------------------------------
const CONF_THRESHOLD = 0.15;     // was 0.35 – lower to see more candidates
const IOU_THRESHOLD  = 0.40;     // was 0.45 – a bit more permissive
const TARGET_FPS     = 24;
const MODEL_NAME     = 'yolov8n.onnx'; // placed at repo root next to index.html
// -------------------------------------------------------------

// DOM refs
const video   = document.getElementById('cam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const hudBox  = document.getElementById('hud');  // small status panel
const btn     = document.getElementById('btn');  // “Activate Scanner”

// Simple HUD logger
const hud = {
  lines: [],
  push(msg) {
    this.lines.unshift(`• ${msg}`);
    this.lines = this.lines.slice(0, 8);
    if (hudBox) hudBox.textContent = this.lines.join('\n');
  },
  ok(msg){ this.push(`${msg} ✓`); },
  warn(msg){ this.push(`${msg} ⚠️`); },
  err(msg){ this.push(`Error: ${msg} ✖`); }
};

let session = null;
let inputName = null;
let inputShape = null;
let running = false;

// Fit canvas to video each frame
function fitCanvas() {
  if (!video.videoWidth) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

// Load the ONNX model with fallbacks: local → raw GH → CDN
async function loadModel() {
  hud.push('Loading YOLOv8 (.onnx)…');

  const urls = [
    // Local (GitHub Pages root)
    MODEL_NAME,
    // Raw GitHub (fallback)
    'https://raw.githubusercontent.com/keirduffy25/V-Scanner/refs/heads/main/yolov8n.onnx',
    // CDN fallback (if you archived to a CDN path, place here)
    // 'https://cdn.your-cdn.example/yolov8n.onnx'
  ];

  let lastErr = null;
  for (const url of urls) {
    try {
      hud.push(`Checking model URL…`);
      console.log('Trying model URL:', url);
      const resp = await fetch(url, { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const arrayBuffer = await resp.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Create ORT session
      hud.push('Initialising ORT session…');
      const opts = {
        executionProviders: ['wasm'], // iPad-safe
        graphOptimizationLevel: 'all'
      };
      const s = await ort.InferenceSession.create(bytes, opts);

      inputName = s.inputNames[0];
      inputShape = s.inputMetadata[inputName].dimensions;
      session = s;

      hud.ok('Model loaded');
      console.log('Model loaded:', { inputName, inputShape });
      return;
    } catch (e) {
      lastErr = e;
      console.warn('Model load failed from', url, e);
    }
  }
  throw lastErr || new Error('All model sources failed');
}

// Ask for camera
async function startCamera() {
  hud.push('Starting camera…');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'environment',
      width: { ideal: 1280 },
      height: { ideal: 720 }
    },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  hud.ok('Camera ready');
}

// Preprocess video frame → tensor
function toTensor() {
  // Expect model shape like [1, 3, 640, 640]
  const [, , netH, netW] = inputShape.map(v => (typeof v === 'number' ? v : 640));
  fitCanvas();

  // Draw current frame scaled to model input on an offscreen canvas
  const off = toTensor.off || (toTensor.off = document.createElement('canvas'));
  off.width = netW; off.height = netH;
  const octx = off.getContext('2d', { willReadFrequently: true });

  // Letterbox-fit
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(netW / vw, netH / vh);
  const dw = Math.round(vw * scale), dh = Math.round(vh * scale);
  const dx = Math.floor((netW - dw) / 2), dy = Math.floor((netH - dh) / 2);

  octx.clearRect(0, 0, netW, netH);
  octx.drawImage(video, 0, 0, vw, vh, dx, dy, dw, dh);

  // Read pixels → CHW float32 0..1
  const imageData = octx.getImageData(0, 0, netW, netH);
  const { data } = imageData;
  const chw = new Float32Array(3 * netW * netH);
  let p = 0, r = 0, g = netW * netH, b = g * 2;
  for (let i = 0; i < data.length; i += 4) {
    chw[r++] = data[i]     / 255; // R
    chw[g++] = data[i + 1] / 255; // G
    chw[b++] = data[i + 2] / 255; // B
    p += 4;
  }
  const tensor = new ort.Tensor('float32', chw, [1, 3, netH, netW]);

  // Save scale info to map boxes back
  const scaleInfo = { netW, netH, vw, vh, dx, dy, dw, dh, scale };
  return { tensor, scaleInfo };
}

// Decode using helper from yolo.min.js (must be loaded before this script)
function decode(outputs, scaleInfo) {
  // Expect model output as a single tensor
  const outName = session.outputNames[0];
  const out = outputs[outName]; // ort.Tensor
  // The helper "YOLO.decode" should exist if yolo.min.js is present
  if (typeof YOLO?.decode !== 'function') {
    throw new Error('yolo.min.js not loaded before script.js');
  }
  return YOLO.decode(out, {
    confThresh: CONF_THRESHOLD,
    iouThresh: IOU_THRESHOLD,
    scaleInfo
  });
}

// Draw boxes & labels
function drawDetections(dets) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = '14px monospace';
  ctx.textBaseline = 'top';

  dets.forEach(d => {
    const { x, y, w, h, label, conf } = d;
    // cyan box
    ctx.strokeStyle = 'rgba(0,255,255,0.9)';
    ctx.shadowColor = 'rgba(0,255,255,0.8)';
    ctx.shadowBlur = 8;
    ctx.strokeRect(x, y, w, h);
    ctx.shadowBlur = 0;

    const text = `${label} ${(conf * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 8;
    const th = 18;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(x, Math.max(0, y - th), tw, th);
    ctx.fillStyle = '#00FFFF';
    ctx.fillText(text, x + 4, Math.max(0, y - th) + 2);
  });
}

// Main loop — runs at ~TARGET_FPS
async function loop() {
  if (!running) return;

  // NEW: wait until we have enough camera data
  if (video.readyState < 2) {
    requestAnimationFrame(loop);
    return;
  }

  const start = performance.now();
  try {
    const { tensor, scaleInfo } = toTensor();

    hud.push('Running inference…');
    console.log('Running YOLO on frame…');

    const outputs = await session.run({ [inputName]: tensor });

    const dets = decode(outputs, scaleInfo) || [];
    console.log('Detections:', dets.length);
    hud.push(`Detections: ${dets.length}`);

    drawDetections(dets);
  } catch (e) {
    hud.err(e.message || String(e));
    console.error(e);
  }

  // simple FPS pacing
  const elapsed = performance.now() - start;
  const delay = Math.max(0, (1000 / TARGET_FPS) - elapsed);
  setTimeout(() => requestAnimationFrame(loop), delay);
}

// Button handler
btn?.addEventListener('click', async () => {
  if (running) return;
  running = true;
  hud.push('V-Scanner initialising…');

  try {
    // quick capability check
    if (!navigator.mediaDevices?.getUserMedia) {
      hud.err('Camera API not supported');
      return;
    }
    hud.push('Model input: images');

    await loadModel();
    await startCamera();

    hud.ok('ORT ready');
    hud.push('Tap Activate Scanner to begin…');
    requestAnimationFrame(loop);
  } catch (e) {
    running = false;
    hud.err(e.message || String(e));
    alert(e.message || String(e));
  }
});

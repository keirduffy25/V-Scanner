/* ===== Cyberpunk V-Scanner — script.js (YOLOv8 ONNX, ORT-Web) ===== */

const els = {
  video: document.getElementById('cam'),
  canvas: document.getElementById('overlay'),
  hud:    document.getElementById('hud'),
  btn:    document.getElementById('toggle')
};

// ---------------- Config ----------------
const MODEL_SOURCES = [
  './yolov8n.onnx', // local (recommended on GitHub Pages)
  // CDN fallbacks (kept as backup; order matters)
  'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
  'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx'
];

const INPUT_SIZE = 640;              // YOLOv8 default
const CONF_THRES = 0.30;             // show detections above this
const IOU_THRES  = 0.45;             // NMS IoU threshold
const SINGLE_BOX = false;            // set true to lock to top detection only

// -------------- Globals ----------------
let session = null;
let running = false;
let rafId = 0;
let lastDetections = [];

const ctx = els.canvas.getContext('2d', { alpha: true });

// Status HUD helpers
function logLine(text, ok = true) {
  if (!els.hud) return;
  const line = document.createElement('div');
  line.textContent = text;
  line.style.opacity = ok ? '1' : '0.9';
  line.className = ok ? 'ok' : 'warn';
  els.hud.appendChild(line);
  // keep panel inside screen even on iPad pinch-zoom
  els.hud.style.left = '12px';
  els.hud.style.top = '12px';
}
function resetHUD() { if (els.hud) els.hud.innerHTML = ''; }

// ---------- Camera ----------
async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'environment',
      width: { ideal: 1280 },
      height:{ ideal: 720 }
    },
    audio: false
  });
  els.video.srcObject = stream;
  await els.video.play();

  // match canvas to the actual video display size
  const vw = els.video.videoWidth;
  const vh = els.video.videoHeight;
  els.canvas.width = vw;
  els.canvas.height = vh;
  logLine('Camera ready ✓');
}

// ---------- Model ----------
async function loadModel() {
  // ORT execution provider: wasm is the only one available on iOS Safari/Chrome
  const ort = window.ort;
  if (!ort) throw new Error('ONNX Runtime web (ort.min.js) not loaded');

  // try sources in order until one loads
  let lastErr = null;
  for (const url of MODEL_SOURCES) {
    try {
      logLine(`Loading YOLOv8 from: ${url.includes('./') ? '(local)…' : '(cdn)…'}`);
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm']
      });
      logLine('Model loaded ✓');
      return;
    } catch (e) {
      lastErr = e;
      console.warn('Model load failed from:', url, e);
    }
  }
  throw lastErr || new Error('Failed to load any model source');
}

// ---------- Preprocess (letterbox to 640) ----------
const work = document.createElement('canvas');
const wctx = work.getContext('2d');

function preprocess(video) {
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;

  // letterbox scale
  const scale = Math.min(INPUT_SIZE / srcW, INPUT_SIZE / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const dx = Math.floor((INPUT_SIZE - newW) / 2);
  const dy = Math.floor((INPUT_SIZE - newH) / 2);

  work.width = INPUT_SIZE;
  work.height = INPUT_SIZE;
  wctx.clearRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  // fill with 114 (YOLO preprocessing gray)
  wctx.fillStyle = 'rgb(114,114,114)';
  wctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  wctx.drawImage(video, 0, 0, srcW, srcH, dx, dy, newW, newH);

  // to CHW float32 [1,3,640,640] normalized 0..1
  const img = wctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const data = img.data;
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  let i = 0, c0 = 0, c1 = INPUT_SIZE * INPUT_SIZE, c2 = c1 * 2;
  for (let y = 0; y < INPUT_SIZE; y++) {
    for (let x = 0; x < INPUT_SIZE; x++) {
      const r = data[i++] / 255;
      const g = data[i++] / 255;
      const b = data[i++] / 255;
      i++; // skip alpha
      const idx = y * INPUT_SIZE + x;
      chw[c0 + idx] = r;
      chw[c1 + idx] = g;
      chw[c2 + idx] = b;
    }
  }

  const tensor = new ort.Tensor('float32', chw, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  return { tensor, scale, dx, dy, srcW, srcH };
}

// ---------- Decode outputs (NEW: objectness × class prob; auto-layout) ----------
function decode(tensor, confThres) {
  const dims = tensor.dims; // e.g. [1, 8400, 84/85] or [1, 84/85, 8400]
  const data = tensor.data;

  // true if shape is [1,8400,84+] (boxes-first)
  const boxesFirst = dims[1] > dims[2];

  // 5 head channels: x y w h obj
  const anchor = 5;

  let num, classes;
  if (boxesFirst) {
    num = dims[1];
    classes = dims[2] - anchor;
  } else {
    num = dims[2];
    classes = dims[1] - anchor;
  }

  const out = [];

  if (boxesFirst) {
    // layout [1, N, 5+C]
    const step = anchor + classes;
    for (let i = 0; i < num; i++) {
      const off = i * step;
      const cx = data[off + 0];
      const cy = data[off + 1];
      const w  = data[off + 2];
      const h  = data[off + 3];
      const obj = data[off + 4];

      let best = -1, conf = 0;
      for (let c = 0; c < classes; c++) {
        const p = obj * data[off + anchor + c]; // IMPORTANT: multiply by objectness
        if (p > conf) { conf = p; best = c; }
      }
      if (conf >= confThres) out.push({ cx, cy, w, h, conf, cls: best });
    }
  } else {
    // layout [1, 5+C, N]
    const N = num;
    const stride = N;
    const cxA = data.subarray(0 * stride, 1 * stride);
    const cyA = data.subarray(1 * stride, 2 * stride);
    const wA  = data.subarray(2 * stride, 3 * stride);
    const hA  = data.subarray(3 * stride, 4 * stride);
    const objA= data.subarray(4 * stride, 5 * stride);

    for (let i = 0; i < N; i++) {
      let best = -1, conf = 0;
      for (let c = 0; c < classes; c++) {
        const p = objA[i] * data[(anchor + c) * stride + i];
        if (p > conf) { conf = p; best = c; }
      }
      if (conf >= confThres) {
        out.push({ cx: cxA[i], cy: cyA[i], w: wA[i], h: hA[i], conf, cls: best });
      }
    }
  }
  return out;
}

// ---------- NMS ----------
function nms(dets, iouThr) {
  dets.sort((a, b) => b.conf - a.conf);
  const kept = [];
  const iou = (a, b) => {
    const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
    const interX1 = Math.max(ax1, bx1), interY1 = Math.max(ay1, by1);
    const interX2 = Math.min(ax2, bx2), interY2 = Math.min(ay2, by2);
    const iw = Math.max(0, interX2 - interX1);
    const ih = Math.max(0, interY2 - interY1);
    const inter = iw * ih;
    const ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return ua <= 0 ? 0 : inter / ua;
  };
  for (const d of dets) {
    let keep = true;
    for (const k of kept) {
      if (iou(d, k) > iouThr) { keep = false; break; }
    }
    if (keep) kept.push(d);
  }
  return kept;
}

// ---------- Drawing ----------
function draw(dets, map, vw, vh) {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

  const drawOne = d => {
    // map model coords back to video pixels (inverse letterbox)
    const x1 = (d.cx - d.w / 2 - map.dx) / map.scale;
    const y1 = (d.cy - d.h / 2 - map.dy) / map.scale;
    const x2 = (d.cx + d.w / 2 - map.dx) / map.scale;
    const y2 = (d.cy + d.h / 2 - map.dy) / map.scale;

    const left   = Math.max(0, Math.min(vw, x1));
    const top    = Math.max(0, Math.min(vh, y1));
    const width  = Math.max(0, Math.min(vw - left, x2 - x1));
    const height = Math.max(0, Math.min(vh - top,  y2 - y1));

    // box
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00FFC8';
    ctx.shadowColor = '#00FFC8';
    ctx.shadowBlur = 10;
    ctx.strokeRect(left, top, width, height);

    // label
    const label = `${Math.round(d.conf * 100)}%`;
    ctx.font = 'bold 16px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textBaseline = 'top';
    const tW = ctx.measureText(label).width + 10;
    const tH = 18 + 6;

    // keep label on-screen
    let lx = left;
    let ly = Math.max(2, top - tH - 4);
    if (lx + tW > vw) lx = vw - tW - 2;

    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(lx, ly, tW, tH);
    ctx.strokeStyle = '#00FFC8';
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, tW, tH);

    ctx.fillStyle = '#00FFC8';
    ctx.fillText(label, lx + 5, ly + 4);
    ctx.shadowBlur = 0;
  };

  if (SINGLE_BOX && dets.length) {
    drawOne(dets[0]);
  } else {
    for (const d of dets) drawOne(d);
  }
}

// ---------- Main loop ----------
async function tick() {
  if (!running) return;

  const { tensor, scale, dx, dy, srcW, srcH } = preprocess(els.video);
  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = tensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const out = results[outputName];

  // decode + nms
  let dets = decode(out, CONF_THRES);
  dets = nms(dets, IOU_THRES);
  lastDetections = dets;

  draw(dets, { scale, dx, dy }, srcW, srcH);

  rafId = requestAnimationFrame(tick);
}

// ---------- UI ----------
async function start() {
  try {
    resetHUD();
    logLine('Scanner running ✓');
    if (!session) await loadModel();
    await initCamera();
    running = true;
    els.btn && (els.btn.textContent = 'Stop');
    tick();
  } catch (e) {
    console.error(e);
    alert(e.message || String(e));
    stop();
  }
}
function stop() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  els.btn && (els.btn.textContent = 'Start');
}

// Button / tap
if (els.btn) {
  els.btn.addEventListener('click', () => (running ? stop() : start()));
} else {
  // fallback: tap anywhere on video to toggle
  els.video.addEventListener('click', () => (running ? stop() : start()));
}

// Auto-start if allowed by user gesture (iOS requires a tap first sometimes)
window.addEventListener('DOMContentLoaded', () => {
  logLine('Camera ready ⏳', true);
  logLine('Model loaded ⏳', true);
  logLine('Tap Start to begin…', true);
});

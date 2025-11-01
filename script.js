/* ============================================
   Cyberpunk V-Scanner — Version 1 (Fixed Boxes)
   Runs YOLOv8 ONNX model in browser with ORT
   ============================================ */

// ---- DOM References ----
const video  = document.getElementById('cam');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const btn    = document.getElementById('toggle');
const hudBox = document.getElementById('hud');

// ---- Model / Runtime Settings ----
const MODEL_W = 640;
const MODEL_H = 640;
const CONF_TH = 0.25;
const IOU_TH  = 0.45;
const MAX_DETS = 50;

const MODEL_SOURCES = [
  './yolov8n.onnx', // local model beside index.html
  'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx'
];

let session = null;
let running = false;
let rafId = 0;

// ---- HUD Helpers ----
function log(msg, ok = true) {
  if (!hudBox) return;
  const p = document.createElement('div');
  p.textContent = msg;
  p.style.color = ok ? '#9ff' : '#ff6';
  hudBox.appendChild(p);
  hudBox.scrollTop = hudBox.scrollHeight;
}
function clearHUD() { if (hudBox) hudBox.innerHTML = ''; }

// ---- Canvas / Video Sizing ----
function resizeOverlayToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width  = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
}

// ---- View rectangle (for correct box alignment) ----
const view = { x: 0, y: 0, w: 0, h: 0 };
function updateViewRect() {
  if (!video.videoWidth || !video.videoHeight) return;
  const rect = video.getBoundingClientRect();
  const vw = rect.width, vh = rect.height;
  const vidAR = video.videoWidth / video.videoHeight;
  const mdlAR = MODEL_W / MODEL_H;
  if (vidAR >= mdlAR) {
    view.h = vh;
    view.w = vh * mdlAR;
    view.x = (vw - view.w) / 2;
    view.y = 0;
  } else {
    view.w = vw;
    view.h = vw / mdlAR;
    view.x = 0;
    view.y = (vh - view.h) / 2;
  }
}

// ---- Camera / ORT ----
async function ensureCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
}

async function loadModel() {
  if (!window.ort) throw new Error('onnxruntime-web not loaded');
  let lastErr = null;
  for (const url of MODEL_SOURCES) {
    try {
      log(`Loading YOLOv8 from: ${url}`);
      const opts = { executionProviders: ['webgpu', 'wasm'] };
      session = await ort.InferenceSession.create(url, opts);
      log('Model loaded ✓');
      return;
    } catch (e) {
      console.warn('Model load failed', url, e);
      lastErr = e;
    }
  }
  throw lastErr || new Error('All model loads failed');
}

function toTensor(imgCanvas) {
  const { width, height } = imgCanvas;
  const ctx2 = imgCanvas.getContext('2d');
  const { data } = ctx2.getImageData(0, 0, width, height);
  const size = width * height;
  const input = new Float32Array(size * 3);
  for (let i = 0; i < size; i++) {
    input[i] = data[i * 4] / 255;           // R
    input[i + size] = data[i * 4 + 1] / 255; // G
    input[i + size * 2] = data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor('float32', input, [1, 3, height, width]);
}

// ---- Letterbox frame for model ----
function letterboxFrameToModel() {
  const off = document.createElement('canvas');
  off.width = MODEL_W;
  off.height = MODEL_H;
  const ox = off.getContext('2d');
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const vidAR = vw / vh;
  const mdlAR = MODEL_W / MODEL_H;

  let dw, dh, dx, dy;
  if (vidAR >= mdlAR) {
    dh = MODEL_H;
    dw = dh * vidAR;
    dx = (MODEL_W - dw) / 2;
    dy = 0;
  } else {
    dw = MODEL_W;
    dh = dw / vidAR;
    dx = 0;
    dy = (MODEL_H - dh) / 2;
  }

  ox.fillStyle = 'black';
  ox.fillRect(0, 0, MODEL_W, MODEL_H);
  ox.drawImage(video, dx, dy, dw, dh);
  return off;
}

// ---- Decode YOLO ----
function decodeYolo(output, confTh) {
  const buf = output.data;
  const dims = output.dims;
  const res = [];

  let N, C;
  if (dims[1] === 84) { // [1,84,N]
    C = 84; N = dims[2];
    for (let i = 0; i < N; i++) {
      const x = buf[0 * N + i];
      const y = buf[1 * N + i];
      const w = buf[2 * N + i];
      const h = buf[3 * N + i];
      let best = 0, cls = -1;
      for (let c = 0; c < 80; c++) {
        const v = buf[(4 + c) * N + i];
        if (v > best) { best = v; cls = c; }
      }
      if (best >= confTh) res.push({ x: x - w / 2, y: y - h / 2, w, h, conf: best, cls });
    }
  } else if (dims[2] === 84) { // [1,N,84]
    N = dims[1];
    for (let i = 0; i < N; i++) {
      const base = i * 84;
      const x = buf[base];
      const y = buf[base + 1];
      const w = buf[base + 2];
      const h = buf[base + 3];
      let best = 0, cls = -1;
      for (let c = 4; c < 84; c++) {
        const v = buf[base + c];
        if (v > best) { best = v; cls = c - 4; }
      }
      if (best >= confTh) res.push({ x: x - w / 2, y: y - h / 2, w, h, conf: best, cls });
    }
  } else {
    console.warn('Unexpected YOLO output dims', dims);
  }
  return res;
}

// ---- NMS ----
function nms(boxes, iouTh, maxKeep) {
  const keep = [];
  boxes.sort((a, b) => b.conf - a.conf);
  const used = new Array(boxes.length).fill(false);
  function iou(a, b) {
    const ax2 = a.x + a.w, ay2 = a.y + a.h;
    const bx2 = b.x + b.w, by2 = b.y + b.h;
    const ix1 = Math.max(a.x, b.x);
    const iy1 = Math.max(a.y, b.y);
    const ix2 = Math.min(ax2, bx2);
    const iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
  }
  for (let i = 0; i < boxes.length; i++) {
    if (used[i]) continue;
    const a = boxes[i];
    keep.push(a);
    if (keep.length >= maxKeep) break;
    for (let j = i + 1; j < boxes.length; j++) {
      if (used[j]) continue;
      const b = boxes[j];
      if (iou(a, b) > iouTh) used[j] = true;
    }
  }
  return keep;
}

// ---- Map model boxes → canvas ----
function mapModelBoxToCanvasXYWH(x, y, w, h) {
  const sx = view.w / MODEL_W;
  const sy = view.h / MODEL_H;
  return {
    x: view.x + x * sx,
    y: view.y + y * sy,
    w: w * sx,
    h: h * sy,
  };
}

// ---- Draw Detections ----
function drawDetections(dets) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resizeOverlayToVideo();
  updateViewRect();

  for (const d of dets) {
    const m = mapModelBoxToCanvasXYWH(d.x, d.y, d.w, d.h);
    const vw = window.innerWidth, vh = window.innerHeight;
    m.x = Math.max(0, Math.min(m.x, vw - 1));
    m.y = Math.max(0, Math.min(m.y, vh - 1));
    m.w = Math.max(1, Math.min(m.w, vw - m.x));
    m.h = Math.max(1, Math.min(m.h, vh - m.y));

    const n = className(d.cls);
    const glow = ['person','cat','dog','horse','cow','elephant','bear','zebra','giraffe'].includes(n)
      ? '#00b3ff' : '#00ff88';

    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = glow;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 12;
    ctx.strokeRect(m.x, m.y, m.w, m.h);

    const label = `${n} ${(d.conf * 100 | 0)}%`;
    ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    const tw = Math.ceil(ctx.measureText(label).width);
    const th = 18;
    let lx = m.x, ly = m.y - th - 4;
    const M = 12;
    if (ly < M) ly = Math.min(m.y + 4, vh - th - M);
    if (lx + tw + 8 > vw - M) lx = vw - tw - 8 - M;
    if (lx < M) lx = M;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(lx - 4, ly, tw + 8, th);
    ctx.fillStyle = glow;
    ctx.fillText(label, lx, ly + th - 5);
    ctx.restore();
  }
}

// ---- Class Names ----
function className(idx) {
  const names = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
  ];
  return names[idx] ?? `cls${idx}`;
}

// ---- Main Loop ----
async function tick() {
  if (!running) return;
  try {
    const frame = letterboxFrameToModel();
    const tensor = toTensor(frame);
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const raw = results[session.outputNames[0]];
    let dets = decodeYolo(raw, CONF_TH);
    dets = nms(dets, IOU_TH, MAX_DETS);
    drawDetections(dets);
  } catch (e) {
    console.error(e);
    log(`❌ ${e.message}`, false);
  }
  rafId = requestAnimationFrame(tick);
}

// ---- Controls ----
async function start() {
  if (running) return;
  clearHUD();
  log('Starting scanner…');
  await ensureCamera();
  resizeOverlayToVideo();
  updateViewRect();
  video.addEventListener('loadedmetadata', () => {
    resizeOverlayToVideo();
    updateViewRect();
  });
  window.addEventListener('resize', () => { resizeOverlayToVideo(); updateViewRect(); });
  window.addEventListener('orientationchange', () => setTimeout(() => { resizeOverlayToVideo(); updateViewRect(); }, 200));
  if (!session) await loadModel();
  running = true;
  if (btn) btn.textContent = 'Stop';
  rafId = requestAnimationFrame(tick);
}

function stop() {
  running = false;
  if (btn) btn.textContent = 'Start';
  cancelAnimationFrame(rafId);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

if (btn) btn.addEventListener('click', () => (running ? stop() : start()));

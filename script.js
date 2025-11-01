/* ===============================
   Cyberpunk V-Scanner — Version 1
   - Runs YOLOv8n ONNX in the browser via onnxruntime-web (ORT)
   - Draws perfectly aligned boxes on an overlay canvas
   - Works on iPad/iPhone (WebKit) and desktop
   - No yolo.min.js dependency (Option B)
   ================================= */

// ---- DOM ----
const video   = document.getElementById('cam');        // <video id="cam">
const canvas  = document.getElementById('overlay');    // <canvas id="overlay">
const ctx     = canvas.getContext('2d');
const btn     = document.getElementById('toggle');     // Start/Stop button (optional)
const hudBox  = document.getElementById('hud');        // Small status HUD (optional)

// ---- Model / ORT config ----
const MODEL_W = 640;
const MODEL_H = 640;
const CONF_TH = 0.25;   // confidence threshold
const IOU_TH  = 0.45;   // NMS IoU threshold
const MAX_DETS = 50;

const MODEL_SOURCES = [
  './yolov8n.onnx', // local beside index.html (GitHub Pages friendly)
  // You can add extra mirrors here if you like:
  // 'https://raw.githubusercontent.com/<you>/<repo>/main/yolov8n.onnx',
];

let session = null;
let running = false;
let rafId = 0;

// ---------- Utilities ----------

function log(msg, ok) {
  if (!hudBox) return;
  const p = document.createElement('div');
  p.textContent = msg;
  p.style.opacity = ok === false ? '0.85' : '1';
  p.style.color = ok === false ? '#ff6' : '#9ff';
  hudBox.appendChild(p);
  hudBox.scrollTop = hudBox.scrollHeight;
}

function clearHUD() { if (hudBox) hudBox.innerHTML = ''; }

// Keep overlay in perfect sync with the video element (CSS + HiDPI buffer)
function resizeOverlayToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;

  const rect = video.getBoundingClientRect();

  // CSS size (matches layout box)
  canvas.style.width  = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  // Internal pixel buffer (retina sharp)
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(rect.width  * dpr);
  canvas.height = Math.round(rect.height * dpr);

  // 1 canvas unit = 1 CSS pixel
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
}

/**
 * Letterbox the current video frame into a MODEL_W x MODEL_H canvas,
 * preserving aspect ratio and padding with black. Returns a {canvas, sx, sy, dx, dy}
 * describing how the model view was placed (used later to map boxes back).
 */
function letterboxFrameToModel() {
  const off = document.createElement('canvas');
  off.width = MODEL_W;
  off.height = MODEL_H;
  const ox = off.getContext('2d');

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const modelAR = MODEL_W / MODEL_H;
  const vidAR   = vw / vh;

  let drawW, drawH, dx, dy;
  if (vidAR >= modelAR) {
    // fit height
    drawH = MODEL_H;
    drawW = Math.round(drawH * vidAR);
    dx = Math.round((MODEL_W - drawW) / 2);
    dy = 0;
  } else {
    // fit width
    drawW = MODEL_W;
    drawH = Math.round(drawW / vidAR);
    dx = 0;
    dy = Math.round((MODEL_H - drawH) / 2);
  }

  // pad background
  ox.fillStyle = 'black';
  ox.fillRect(0, 0, MODEL_W, MODEL_H);

  ox.drawImage(video, dx, dy, drawW, drawH);

  // scale factors from model space back to “letterboxed view” region
  const sx = drawW / MODEL_W;
  const sy = drawH / MODEL_H;

  return { canvas: off, sx, sy, dx, dy, drawW, drawH };
}

/**
 * Map a box from model space (0..640) into the overlay canvas,
 * respecting the letterboxed placement inside the video element.
 * Input box uses top-left + size: {x,y,w,h} in MODEL space.
 */
function mapBoxToCanvas(box, letterboxInfo) {
  const rect = video.getBoundingClientRect();   // CSS pixels
  const vw = rect.width;
  const vh = rect.height;

  // Compute where the model view sits inside the video CSS box
  const videoAR = video.videoWidth / video.videoHeight;
  const modelAR = MODEL_W / MODEL_H;

  let viewW, viewH, offX, offY;
  if (videoAR >= modelAR) {
    viewH = vh;
    viewW = vh * modelAR;
    offX = (vw - viewW) / 2;
    offY = 0;
  } else {
    viewW = vw;
    viewH = vw / modelAR;
    offX = 0;
    offY = (vh - viewH) / 2;
  }

  const sx = viewW / MODEL_W;
  const sy = viewH / MODEL_H;

  return {
    x: offX + box.x * sx,
    y: offY + box.y * sy,
    w: box.w * sx,
    h: box.h * sy,
  };
}

// Basic NMS
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

// Decode YOLOv8 ONNX output tensor to boxes in MODEL space
// Assumes output shape [1, N, 84] OR [1, 84, N]. Handles both.
function decodeYolo(output, confTh) {
  // Support both layouts
  let data, rows, cols;
  const out = output.data;
  const dims = output.dims; // e.g. [1, 84, 8400] or [1, 8400, 84]

  if (dims[1] === 84) {
    // [1, N, 84]
    rows = dims[1]; // N
    // Actually this indicates N=84 which is wrong; check dims carefully:
    // safer path:
  }

  let N, C;
  if (dims[1] === 84) {          // [1, 84, N]
    C = 84; N = dims[2];
    // transpose to [N,84] on the fly
    return decodeChannels84N(out, N, confTh);
  } else if (dims[2] === 84) {   // [1, N, 84]
    N = dims[1]; C = 84;
    return decodeN84(out, N, confTh);
  } else {
    console.warn('Unexpected YOLO output dims:', dims);
    return [];
  }

  // Helpers
  function decodeN84(buf, N, th) {
    const res = [];
    for (let i = 0; i < N; i++) {
      const base = i * 84;
      const x = buf[base + 0];
      const y = buf[base + 1];
      const w = buf[base + 2];
      const h = buf[base + 3];
      let best = 0, bestIdx = -1;
      for (let c = 4; c < 84; c++) {
        const v = buf[base + c];
        if (v > best) { best = v; bestIdx = c - 4; }
      }
      const conf = best;
      if (conf >= th) {
        res.push({ x: x - w/2, y: y - h/2, w, h, conf, cls: bestIdx });
      }
    }
    return res;
  }

  function decodeChannels84N(buf, N, th) {
    const res = [];
    // First 4*N are boxes (cx,cy,w,h), remaining are class logits/scores
    for (let i = 0; i < N; i++) {
      const x = buf[0*N + i];
      const y = buf[1*N + i];
      const w = buf[2*N + i];
      const h = buf[3*N + i];
      let best = 0, bestIdx = -1;
      for (let c = 0; c < 80; c++) {
        const v = buf[(4 + c)*N + i];
        if (v > best) { best = v; bestIdx = c; }
      }
      const conf = best;
      if (conf >= th) {
        res.push({ x: x - w/2, y: y - h/2, w, h, conf, cls: bestIdx });
      }
    }
    return res;
  }
}

// Draw aligned boxes and label tabs
function drawDetections(dets) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resizeOverlayToVideo();

  for (const d of dets) {
    const mapped = mapBoxToCanvas(d);

    // Box
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(57,255,20,0.95)';
    ctx.strokeRect(mapped.x, mapped.y, mapped.w, mapped.h);

    // Label
    const label = `${className(d.cls)} ${(d.conf * 100) | 0}%`;
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    const padX = 6, padY = 4;
    const tw = ctx.measureText(label).width;

    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(mapped.x, mapped.y - 16, tw + padX * 2, 16);
    ctx.fillStyle = 'rgba(57,255,20,1)';
    ctx.fillText(label, mapped.x + padX, mapped.y - 4);
  }
}

function className(idx) {
  // COCO 80 classes (trimmed list for brevity; add the rest if you like)
  const names = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
  ];
  return names[idx] ?? `cls${idx}`;
}

// ---------- Camera + ORT ----------

async function ensureCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

async function loadModel() {
  if (!window.ort) throw new Error('onnxruntime-web not loaded. Include <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script> before script.js');

  // Prefer WebGPU for speed (where supported), else wasm
  let providers = [];
  if (ort.env.webgpu) providers.push('webgpu');
  providers.push('wasm');

  let lastErr = null;
  for (const url of MODEL_SOURCES) {
    try {
      log(`Loading YOLOv8 from: ${url}`);
      const opt = { executionProviders: providers };
      session = await ort.InferenceSession.create(url, opt);
      log('Model loaded ✓');
      return;
    } catch (e) {
      lastErr = e;
      console.warn('Model load failed from', url, e);
    }
  }
  throw lastErr || new Error('All model sources failed');
}

function toTensor(imgCanvas) {
  const { width, height } = imgCanvas;
  const ctx2 = imgCanvas.getContext('2d');
  const { data } = ctx2.getImageData(0, 0, width, height); // RGBA
  const size = width * height;
  const input = new Float32Array(size * 3);
  // Convert to NCHW normalized [0,1]
  for (let i = 0; i < size; i++) {
    const r = data[i * 4 + 0] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    input[i] = r;
    input[i + size] = g;
    input[i + size * 2] = b;
  }
  return new ort.Tensor('float32', input, [1, 3, height, width]);
}

// Main loop
async function tick() {
  if (!running) return;

  try {
    // Prepare model input (letterboxed)
    const letter = letterboxFrameToModel();
    const tensor = toTensor(letter.canvas);

    const feeds = {};
    const inputName = session.inputNames[0];
    feeds[inputName] = tensor;

    const results = await session.run(feeds);
    const outName = session.outputNames[0];
    const raw = results[outName]; // ort.Tensor

    // Decode → NMS → draw
    let dets = decodeYolo(raw, CONF_TH);
    dets = nms(dets, IOU_TH, MAX_DETS);

    drawDetections(dets);
  } catch (e) {
    console.error(e);
    log(`❌ ${e.message || e}`, false);
  }

  rafId = requestAnimationFrame(tick);
}

// ---------- Controls ----------

async function start() {
  if (running) return;
  clearHUD();
  log('V-Scanner initialising…');
  await ensureCamera();
  resizeOverlayToVideo();

  // resize when metadata is ready + on future resizes/rotations
  video.addEventListener('loadedmetadata', resizeOverlayToVideo, { once: true });
  window.addEventListener('resize', resizeOverlayToVideo);
  window.addEventListener('orientationchange', () => setTimeout(resizeOverlayToVideo, 200));

  log('Camera ready ✓');
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

// Optional button wiring
if (btn) {
  btn.addEventListener('click', () => (running ? stop() : start()));
}

// Auto-start if you prefer:
// start();

// Expose for console debugging
window.VScanner = { start, stop, resizeOverlayToVideo };

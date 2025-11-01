/* =========================
   Cyberpunk V-Scanner v1 (local YOLOv8n.onnx)
   - iPad-safe overlay sizing (DPR aware)
   - Confidence filter + NMS + single-target focus
   - Works with onnxruntime-web (WASM)
   ========================= */

const video  = document.getElementById('cam');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const hudCount = document.getElementById('hud-count');
const btn = document.getElementById('toggleBtn');

let session = null;
let running = false;
let rafId   = null;

// ---------- Canvas fit (HiDPI + resize/orientation) ----------
function fitCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;

  // CSS size
  canvas.style.width  = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  // Backing pixels
  canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

  // draw in CSS px
  ctx.setTransform(1,0,0,1,0,0);
  ctx.scale(dpr, dpr);
}
const ro = new ResizeObserver(fitCanvasToVideo);
ro.observe(video);
window.addEventListener('orientationchange', () => setTimeout(fitCanvasToVideo, 250));
video.addEventListener('loadedmetadata', fitCanvasToVideo);

// ---------- Camera ----------
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'environment',
      width: { ideal: 1280 },
      height:{ ideal: 720 }
    },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  fitCanvasToVideo();
}

// ---------- YOLO model (local file) ----------
const MODEL_INPUT = 640;         // YOLOv8 default square
const CONF_THRESHOLD = 0.45;     // raise to reduce clutter
const IOU_THRESHOLD  = 0.45;
const MAX_BOXES      = 30;
const SHOW_TOP_K     = 1;        // keep 1 for single-target focus

async function loadModel() {
  // Expect yolov8n.onnx beside index.html (GitHub Pages repo root)
  session = await ort.InferenceSession.create('./yolov8n.onnx', {
    executionProviders: ['wasm']
  });
}

// ---------- Letterbox & preprocess to 640x640 ----------
function letterbox(source, dst, color=[114,114,114]) {
  // draw source into dst (canvas) with padding, maintaining aspect
  const sw = source.videoWidth  || source.width;
  const sh = source.videoHeight || source.height;
  const dw = dst.width;
  const dh = dst.height;

  const r = Math.min(dw / sw, dh / sh);
  const nw = Math.round(sw * r);
  const nh = Math.round(sh * r);
  const dx = Math.floor((dw - nw) / 2);
  const dy = Math.floor((dh - nh) / 2);

  const dctx = dst.getContext('2d');
  dctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
  dctx.fillRect(0, 0, dw, dh);
  dctx.drawImage(source, 0, 0, sw, sh, dx, dy, nw, nh);

  return { scale: r, padX: dx, padY: dy, srcW: sw, srcH: sh };
}

const prepCanvas = document.createElement('canvas');
prepCanvas.width = MODEL_INPUT;
prepCanvas.height= MODEL_INPUT;

// Convert canvas RGBA to Float32 NHWC (0..1) then to NCHW
function toTensorFromCanvas(cnv) {
  const imgData = cnv.getContext('2d').getImageData(0,0,cnv.width,cnv.height).data;
  const N = cnv.width * cnv.height;
  const chw = new Float32Array(3 * N);

  for (let i=0, p=0; i<N; i++, p+=4) {
    const r = imgData[p]   / 255;
    const g = imgData[p+1] / 255;
    const b = imgData[p+2] / 255;
    chw[i]          = r;         // R
    chw[i + N]      = g;         // G
    chw[i + 2 * N]  = b;         // B
  }
  return new ort.Tensor('float32', chw, [1, 3, cnv.height, cnv.width]);
}

// ---------- Postprocess ----------
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

function nms(boxes, thr = IOU_THRESHOLD) {
  const out = [];
  const arr = boxes.slice().sort((a,b)=> b.score - a.score);
  while (arr.length) {
    const pick = arr.shift();
    out.push(pick);
    for (let i=arr.length-1;i>=0;i--) {
      if (iou(pick, arr[i]) > thr) arr.splice(i,1);
    }
  }
  return out;
}

// Map model coords (letterboxed) back to *video pixels*.
function modelBoxToVideo(b, meta) {
  // model gave us xywh in model space (after letterbox)
  const mx = b.x - meta.padX;
  const my = b.y - meta.padY;
  const sx = mx / meta.scale;
  const sy = my / meta.scale;
  const sw = b.w / meta.scale;
  const sh = b.h / meta.scale;
  // clamp
  return {
    x: Math.max(0, Math.min(meta.srcW - 1, sx)),
    y: Math.max(0, Math.min(meta.srcH - 1, sy)),
    w: Math.max(1, Math.min(meta.srcW, sw)),
    h: Math.max(1, Math.min(meta.srcH, sh)),
    score: b.score,
    cls: b.cls
  };
}

// Convert YOLOv8 (1,84,8400) style to array of boxes (xywh conf/cls)
function parseYolo(output, meta) {
  // output is the first entry of session output
  // shape: [1, 84, N]  where 84 = 4 box + 1 obj + 80 classes (COCO)
  const data = output.data;
  const dims = output.dims; // [1,84,N]
  const C = dims[1];
  const N = dims[2];

  const boxes = [];
  for (let i=0; i<N; i++) {
    const x = data[0 * N + i];
    const y = data[1 * N + i];
    const w = data[2 * N + i];
    const h = data[3 * N + i];
    const obj = data[4 * N + i];

    // class scores start at 5
    let best = 5, bestScore = 0;
    for (let c=5; c<C; c++) {
      const sc = data[c * N + i];
      if (sc > bestScore) { bestScore = sc; best = c; }
    }
    const score = sigmoid(obj) * sigmoid(bestScore);
    if (score < CONF_THRESHOLD) continue;

    // Model returns center-x/y, width/height in model space
    const left = x - w/2;
    const top  = y - h/2;

    boxes.push(modelBoxToVideo({
      x: left, y: top, w, h, score, cls: `c${best-5}`
    }, meta));
  }

  // NMS + limit + top-K
  let filtered = nms(boxes).slice(0, MAX_BOXES);
  if (SHOW_TOP_K > 0) filtered = filtered.slice(0, SHOW_TOP_K);
  return filtered;
}

// ---------- Drawing ----------
function modelToCanvas(box) {
  const vw = video.videoWidth  || canvas.width;
  const vh = video.videoHeight || canvas.height;
  const rect = video.getBoundingClientRect();
  const sx = rect.width / vw;
  const sy = rect.height/ vh;
  return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy, score: box.score, cls: box.cls };
}

function drawBoxes(boxes) {
  fitCanvasToVideo();
  ctx.clearRect(0,0,canvas.width,canvas.height);

  ctx.save();
  ctx.lineWidth = 2;
  ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.textBaseline = 'top';

  boxes.forEach(b => {
    const c = modelToCanvas(b);
    ctx.strokeStyle = 'rgba(80,255,200,0.95)';
    ctx.shadowColor = 'rgba(80,255,200,0.6)';
    ctx.shadowBlur  = 8;
    ctx.strokeRect(c.x, c.y, c.w, c.h);

    const label = `${b.cls} ${Math.round(b.score*100)}%`;
    const pad=4, th=16, tw=ctx.measureText(label).width;
    ctx.fillStyle='rgba(0,0,0,0.75)';
    ctx.fillRect(c.x, c.y-(th+6), tw+pad*2, th+6);
    ctx.fillStyle='rgba(170,255,230,0.95)';
    ctx.fillText(label, c.x+pad, c.y-(th+3));
  });

  ctx.restore();
  hudCount.textContent = boxes.length.toString();
}

// ---------- Main loop ----------
async function tick() {
  if (!running) return;

  // 1) letterbox video -> 640x640
  const meta = letterbox(video, prepCanvas);

  // 2) tensor
  const input = toTensorFromCanvas(prepCanvas);

  // 3) run
  const feeds = {};
  const inputName = session.inputNames[0];
  feeds[inputName] = input;
  const output = await session.run(feeds);
  const outName = session.outputNames[0];
  const y = output[outName];

  // 4) parse -> boxes
  const boxes = parseYolo(y, meta);

  // 5) draw
  drawBoxes(boxes);

  rafId = requestAnimationFrame(tick);
}

// ---------- UI ----------
btn.addEventListener('click', async () => {
  if (!running) {
    btn.textContent = 'Stop';
    running = true;
    rafId && cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  } else {
    running = false;
    btn.textContent = 'Start';
    rafId && cancelAnimationFrame(rafId);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    hudCount.textContent = '0';
  }
});

// ---------- Boot ----------
(async () => {
  try {
    await startCamera();
    await loadModel();
  } catch (e) {
    alert('Init error: ' + (e?.message || e));
    console.error(e);
  }
})();

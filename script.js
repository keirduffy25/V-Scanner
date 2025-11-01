/* =========================
   Cyberpunk V-Scanner v1 (robust YOLOv8 ONNX)
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

  canvas.style.width  = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

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
    video: { facingMode: 'environment', width: { ideal: 1280 }, height:{ ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  fitCanvasToVideo();
}

// ---------- YOLO model (local file) ----------
const MODEL_INPUT = 640;
let   CONF_THRESHOLD = 0.25;   // lowered for validation
const IOU_THRESHOLD  = 0.45;
const MAX_BOXES      = 30;
const SHOW_TOP_K     = 1;

async function loadModel() {
  session = await ort.InferenceSession.create('./yolov8n.onnx', {
    executionProviders: ['wasm']
  });
}

// ---------- Letterbox -> 640x640 ----------
function letterbox(source, dst, color=[114,114,114]) {
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

function toTensorFromCanvas(cnv) {
  const imgData = cnv.getContext('2d').getImageData(0,0,cnv.width,cnv.height).data;
  const N = cnv.width * cnv.height;
  const chw = new Float32Array(3 * N);

  for (let i=0, p=0; i<N; i++, p+=4) {
    const r = imgData[p]   / 255;
    const g = imgData[p+1] / 255;
    const b = imgData[p+2] / 255;
    chw[i]          = r;
    chw[i + N]      = g;
    chw[i + 2 * N]  = b;
  }
  return new ort.Tensor('float32', chw, [1, 3, cnv.height, cnv.width]);
}

// ---------- Postprocess helpers ----------
function sigmoid(x){ return 1/(1+Math.exp(-x)); }
function maybeSigmoid(v) {
  // If values look like logits (outside [0,1]), apply sigmoid.
  // Many exports already return [0..1]; this keeps both working.
  return (v < 0 || v > 1) ? sigmoid(v) : v;
}
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
function modelBoxToVideo(b, meta) {
  const mx = b.x - meta.padX;
  const my = b.y - meta.padY;
  const sx = mx / meta.scale;
  const sy = my / meta.scale;
  const sw = b.w / meta.scale;
  const sh = b.h / meta.scale;
  return {
    x: Math.max(0, Math.min(meta.srcW - 1, sx)),
    y: Math.max(0, Math.min(meta.srcH - 1, sy)),
    w: Math.max(1, Math.min(meta.srcW, sw)),
    h: Math.max(1, Math.min(meta.srcH, sh)),
    score: b.score,
    cls: b.cls
  };
}

// ---------- Robust YOLOv8 parser (handles [1,84,N] and [1,N,84]) ----------
function parseYolo(output, meta) {
  const data = output.data;
  const dims = output.dims; // expect [1,84,N] or [1,N,84]

  let C, N, layout; // layout: 'CN' => [1,84,N], 'NC' => [1,N,84]
  if (dims.length === 3 && dims[0] === 1) {
    if (dims[1] === 84) { C = 84; N = dims[2]; layout = 'CN'; }
    else if (dims[2] === 84) { C = 84; N = dims[1]; layout = 'NC'; }
    else {
      console.warn('Unexpected output dims', dims);
      return [];
    }
  } else {
    console.warn('Unexpected output dims', dims);
    return [];
  }

  const boxes = [];
  for (let i=0; i<N; i++) {
    // pull fields depending on layout
    let x,y,w,h,obj; let clsStartIdx;
    if (layout === 'CN') {
      x   = data[0 * N + i];
      y   = data[1 * N + i];
      w   = data[2 * N + i];
      h   = data[3 * N + i];
      obj = data[4 * N + i];
      clsStartIdx = 5 * N + i;
    } else {
      const base = i * C;
      x   = data[base + 0];
      y   = data[base + 1];
      w   = data[base + 2];
      h   = data[base + 3];
      obj = data[base + 4];
      clsStartIdx = base + 5;
    }

    // best class
    let bestClass = 0, bestScoreRaw = -1e9;
    for (let c = 0; c < 80; c++) {
      const v = (layout === 'CN') ? data[clsStartIdx + c * N] : data[clsStartIdx + c];
      if (v > bestScoreRaw) { bestScoreRaw = v; bestClass = c; }
    }

    const objP = maybeSigmoid(obj);
    const clsP = maybeSigmoid(bestScoreRaw);
    const score = objP * clsP;
    if (score < CONF_THRESHOLD) continue;

    // xywh are center format in model pixels (letterboxed 640x640)
    const left = x - w/2;
    const top  = y - h/2;

    boxes.push(modelBoxToVideo({
      x: left, y: top, w, h, score, cls: cocoName(bestClass)
    }, meta));
  }

  let filtered = nms(boxes).slice(0, MAX_BOXES);
  if (SHOW_TOP_K > 0) filtered = filtered.slice(0, SHOW_TOP_K);
  return filtered;
}

// COCO class names (short)
const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];
function cocoName(i){ return COCO[i] ?? `c${i}`; }

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

  const meta  = letterbox(video, prepCanvas);
  const input = toTensorFromCanvas(prepCanvas);

  const feeds = {};
  feeds[session.inputNames[0]] = input;

  const output = await session.run(feeds);
  const out = output[session.outputNames[0]];

  const boxes = parseYolo(out, meta);
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

/*  V-Scanner — YOLOv8n (ONNX + ONNX Runtime Web)
 *  Version: 1.4  (fixes iPad top-left clipping by mapping in video pixel space)
 *  Files expected next to index.html:  script.js, style.css, yolov8n.onnx
 *  index.html must include onnxruntime-web:
 *    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js"></script>
 */

const MODEL_W = 640;
const MODEL_H = 640;
const SCORE_TH = 0.25;   // confidence threshold
const NMS_IOU = 0.45;

const hud = document.getElementById('hud') || makeHud();
const video = document.getElementById('cam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const btn = document.getElementById('toggle');

let session = null;
let scanning = false;
let rafId = 0;
let viewRect = {x:0,y:0,w:0,h:0};
let off = document.createElement('canvas'); // preprocess canvas
let offctx = off.getContext('2d', { willReadFrequently: true });

// COCO 80 classes
const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

function makeHud() {
  const d = document.createElement('div');
  d.id = 'hud';
  d.style.cssText = 'position:fixed;top:8px;left:8px;z-index:20;font:12px ui-monospace,Menlo,Consolas,monospace;color:#9ff;background:rgba(0,0,0,.35);padding:6px 8px;border:1px solid rgba(0,255,255,.25);border-radius:6px;backdrop-filter:blur(4px);white-space:nowrap;';
  document.body.appendChild(d);
  return d;
}
function hudLine(t, ok=true){ hud.innerText += (hud.innerText?'\n':'') + (ok?'• ':'× ') + t; }
function hudReset(){ hud.innerText=''; }

function className(i){ return COCO[i|0] ?? `id:${i}`; }

// ---------- layout helpers ----------
function resizeOverlayToVideo() {
  // Sync drawing canvas to CSS pixels (then scaled by DPR for crisp lines)
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = video.getBoundingClientRect();
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';
  canvas.width  = Math.round(rect.width  * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function updateViewRect() {
  // cache current drawing area in CSS pixels
  const r = canvas.getBoundingClientRect();
  viewRect = { x:r.left, y:r.top, w:r.width, h:r.height };
}

// ---------- mapping & drawing (FIXED) ----------
function mapModelBoxToCanvasXYWH(x, y, w, h) {
  // Convert YOLO model-space box (640×640) to current canvas pixels.
  // Uses canvas CSS size only; stable on iPad Safari/Chrome.
  const rect = canvas.getBoundingClientRect();
  const vw = rect.width;
  const vh = rect.height;

  const sx = vw / MODEL_W;
  const sy = vh / MODEL_H;

  // If your <video> is letterboxed within the rect, center offsets protect alignment.
  // We assume the rendered video matches MODEL_AR after object-fit: cover/contain handling.
  const mdlAR = MODEL_W / MODEL_H; // 1:1 for 640×640
  const vidAR = vw / vh;
  let offsetX = 0, offsetY = 0;
  if (vidAR > mdlAR) {
    const scaledW = vh * mdlAR;
    offsetX = (vw - scaledW) / 2;
  } else if (vidAR < mdlAR) {
    const scaledH = vw / mdlAR;
    offsetY = (vh - scaledH) / 2;
  }
  return {
    x: offsetX + x * sx,
    y: offsetY + y * sy,
    w: w * sx,
    h: h * sy
  };
}

function drawDetections(dets) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resizeOverlayToVideo();
  updateViewRect();

  const rect = canvas.getBoundingClientRect();
  const vw = rect.width;
  const vh = rect.height;

  for (const d of dets) {
    const m = mapModelBoxToCanvasXYWH(d.x, d.y, d.w, d.h);
    const name = className(d.cls);
    const glow = ['person','cat','dog','horse','cow','elephant','bear','zebra','giraffe'].includes(name)
      ? '#00b3ff' : '#00ff88';

    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = glow;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 12;
    ctx.strokeRect(m.x, m.y, m.w, m.h);

    const label = `${name} ${(d.conf*100|0)}%`;
    ctx.font = '13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    const tw = Math.ceil(ctx.measureText(label).width);
    const th = 18;
    const M = 12;
    let lx = m.x, ly = m.y - th - 4;
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

// ---------- camera ----------
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment',
        width: { ideal: 1280 },
        height:{ ideal: 720 }
      },
      audio: false
    });
    video.srcObject = stream;
    video.setAttribute('playsinline','');
    video.muted = true;
    await video.play().catch(()=>{});
    await new Promise(r => {
      if (video.readyState >= 2) return r();
      video.onloadedmetadata = () => r();
    });
    hudLine('Camera ready ✓', true);
  } catch (e) {
    hudLine('Camera failed: ' + e.message, false);
    throw e;
  }
}

// ---------- model load (local first, then raw.githubusercontent fallback) ----------
async function loadModel() {
  const urls = [
    './yolov8n.onnx',
    'https://raw.githubusercontent.com/keirduffy25/V-Scanner/refs/heads/main/yolov8n.onnx'
  ];
  let lastErr = null;
  for (const url of urls) {
    try {
      hudLine(`Loading YOLOv8 from: ${url.includes('http') ? '(cdn)' : '(local)'}…`);
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      hudLine('Model loaded ✓', true);
      return;
    } catch (e) {
      lastErr = e;
      console.warn('Model load failed from', url, e);
    }
  }
  hudLine('Model load error', false);
  throw lastErr;
}

// ---------- preprocessing ----------
function toNCHWFloat32(imageBitmapOrVideo) {
  // Letterbox into 640×640 (black bars), normalize to [0,1], NCHW
  off.width = MODEL_W;
  off.height = MODEL_H;

  const iw = imageBitmapOrVideo.videoWidth || imageBitmapOrVideo.width;
  const ih = imageBitmapOrVideo.videoHeight || imageBitmapOrVideo.height;
  const iAR = iw/ih;
  const mAR = MODEL_W/MODEL_H;

  let dw=MODEL_W, dh=MODEL_H, dx=0, dy=0;
  if (iAR > mAR) { // wider than model
    dh = Math.round(MODEL_W / iAR);
    dy = Math.floor((MODEL_H - dh)/2);
  } else if (iAR < mAR) {
    dw = Math.round(MODEL_H * iAR);
    dx = Math.floor((MODEL_W - dw)/2);
  }

  offctx.clearRect(0,0,MODEL_W,MODEL_H);
  offctx.fillStyle = '#000';
  offctx.fillRect(0,0,MODEL_W,MODEL_H);
  offctx.drawImage(imageBitmapOrVideo, 0,0, iw,ih, dx,dy, dw,dh);

  const { data } = offctx.getImageData(0,0,MODEL_W,MODEL_H);
  const out = new Float32Array(3 * MODEL_H * MODEL_W);
  let p=0, c0=0, c1=MODEL_W*MODEL_H, c2=c1 + MODEL_W*MODEL_H;
  for (let i=0;i<data.length;i+=4) {
    const r = data[i]   /255;
    const g = data[i+1] /255;
    const b = data[i+2] /255;
    out[c0++] = r;
    out[c1++] = g;
    out[c2++] = b;
    p+=4;
  }
  return out;
}

// ---------- postprocess ----------
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function xywh2xyxy(x,y,w,h){
  const x0 = x - w/2;
  const y0 = y - h/2;
  return [x0,y0, x0+w, y0+h];
}

function iou(a,b){
  const x1=Math.max(a[0],b[0]);
  const y1=Math.max(a[1],b[1]);
  const x2=Math.min(a[2],b[2]);
  const y2=Math.min(a[3],b[3]);
  const iw=Math.max(0,x2-x1), ih=Math.max(0,y2-y1);
  const inter=iw*ih;
  const ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
  return ua<=0 ? 0 : inter/ua;
}

function nms(boxes, scores, iouTh) {
  const idxs = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const keep=[];
  for (const i of idxs) {
    let ok=true;
    for (const j of keep) {
      if (iou(boxes[i], boxes[j]) > iouTh) { ok=false; break; }
    }
    if (ok) keep.push(i);
  }
  return keep;
}

function parseYOLO(output) {
  // Supports [1,84,8400] or [1,8400,84]
  const key = Object.keys(output)[0];
  const out = output[key];
  const data = out.data;
  const dims = out.dims; // e.g., [1,84,8400]
  let C, N; // C=84, N=8400
  let transposed = false;
  if (dims[1] === 84) { C = 84; N = dims[2]; transposed = true; }
  else { C = 84; N = dims[1]; transposed = false; }

  const boxes = [];
  const scores = [];
  const classes = [];
  const xyxys = [];

  for (let i=0;i<N;i++){
    const idx = transposed ? i : i*C;
    const x = data[ transposed ? (0*N + i) : (idx + 0) ];
    const y = data[ transposed ? (1*N + i) : (idx + 1) ];
    const w = data[ transposed ? (2*N + i) : (idx + 2) ];
    const h = data[ transposed ? (3*N + i) : (idx + 3) ];

    let best=-1, bestScore=0;
    for (let c=4;c<84;c++){
      const v = data[ transposed ? (c*N + i) : (idx + c) ];
      if (v>bestScore){ bestScore=v; best=c-4; }
    }
    const conf = sigmoid(bestScore);
    if (conf < SCORE_TH) continue;

    const bb = xywh2xyxy(x,y,w,h);
    xyxys.push(bb);
    boxes.push([x,y,w,h]);
    scores.push(conf);
    classes.push(best);
  }

  const keep = nms(xyxys, scores, NMS_IOU);
  const dets=[];
  for (const k of keep) {
    const [x,y,w,h]=boxes[k];
    dets.push({ x, y, w, h, conf:scores[k], cls:classes[k] });
  }
  return dets;
}

// ---------- main loop ----------
async function tick() {
  if (!scanning) return;
  try {
    const input = toNCHWFloat32(video);
    const tensor = new ort.Tensor('float32', input, [1,3,MODEL_H,MODEL_W]);
    const output = await session.run({ images: tensor }).catch(async () => {
      // Some ONNX exports use 'input' as the name
      return session.run({ input: tensor });
    });
    const dets = parseYOLO(output);
    drawDetections(dets);
  } catch (e) {
    // keep running even if a frame fails
    console.warn('tick error', e);
  }
  rafId = requestAnimationFrame(tick);
}

// ---------- controls ----------
async function startScanner() {
  if (scanning) return;
  hudReset();
  hudLine('Scanner running ✓');
  await initCamera();
  resizeOverlayToVideo();
  updateViewRect();
  if (!session) await loadModel();
  scanning = true;
  btn?.classList.add('on');
  btn && (btn.textContent = 'Stop');
  rafId = requestAnimationFrame(tick);
}

function stopScanner() {
  scanning = false;
  cancelAnimationFrame(rafId);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  btn?.classList.remove('on');
  btn && (btn.textContent = 'Start');
}

// ---------- init ----------
window.addEventListener('resize', () => {
  if (!video) return;
  resizeOverlayToVideo();
  updateViewRect();
});

document.addEventListener('DOMContentLoaded', () => {
  resizeOverlayToVideo();
  updateViewRect();
  if (btn) {
    btn.addEventListener('click', () => scanning ? stopScanner() : startScanner());
  }
  // Optional: auto-start after first user gesture anywhere
  document.body.addEventListener('touchend', oneShotStart, { once:true });
  document.body.addEventListener('click',    oneShotStart, { once:true });
  function oneShotStart(){ if (!scanning) startScanner(); }
});

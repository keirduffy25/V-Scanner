// ===== Cyberpunk V-Scanner (Option B: Pure ONNX Runtime Web) =====
// Files required in repo root (same folder as index.html):
//   - index.html, style.css, script.js, yolov8n.onnx

// ---- DOM ----
const video   = document.getElementById('cam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d', { alpha: true, desynchronized: true });
const btnGo   = document.getElementById('activate');
const btnStop = document.getElementById('stop');

// ---- DPI-safe canvas ----
function resizeCanvas() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2); // cap for perf
  canvas.width  = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // draw in CSS pixels
  ctx.clearRect(0, 0, w, h);
}
window.addEventListener('resize', resizeCanvas, { passive: true });
resizeCanvas();

// ---- HUD helpers (clamped) ----
const HUD_MARGIN = 12;
const statusLines = [];
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
function clampRect(x, y, w, h, vw = window.innerWidth, vh = window.innerHeight) {
  x = clamp(x, 0, Math.max(0, vw - w));
  y = clamp(y, 0, Math.max(0, vh - h));
  w = Math.min(w, vw - x);
  h = Math.min(h, vh - y);
  return { x, y, w, h };
}
function setStatus(...lines){ statusLines.splice(0, statusLines.length, ...lines); }
function drawHUD(lines = statusLines) {
  const vw = window.innerWidth, vh = window.innerHeight;
  const pad = 8, rowH = 16;
  const boxW = Math.min(280, vw - 2 * HUD_MARGIN);
  const boxH = Math.min((lines.length * rowH) + pad * 2, vh - 2 * HUD_MARGIN);
  const x = HUD_MARGIN, y = HUD_MARGIN;

  ctx.save();
  ctx.fillStyle = "rgba(0, 255, 255, 0.08)";
  ctx.strokeStyle = "rgba(0, 255, 255, 0.9)";
  ctx.lineWidth = 1.5;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, boxW, boxH, 8); ctx.fill(); ctx.stroke(); }
  else { ctx.fillRect(x, y, boxW, boxH); ctx.strokeRect(x, y, boxW, boxH); }

  ctx.fillStyle = "#9ff";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  let ty = y + pad + 12;
  for (const s of lines) { ctx.fillText(s, x + pad, ty); ty += rowH; }
  ctx.restore();
}

function drawDetBox(x, y, w, h, label, score, glow = '#00fff2') {
  const vw = window.innerWidth, vh = window.innerHeight;
  ({ x, y, w, h } = clampRect(x, y, w, h, vw, vh));

  ctx.save();
  ctx.strokeStyle = glow; ctx.shadowColor = glow; ctx.shadowBlur = 12;
  ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);

  const txt = `${label} ${(score * 100 | 0)}%`;
  ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  const tw = Math.ceil(ctx.measureText(txt).width);
  const th = 18;
  let lx = x, ly = y - th - 4;         // try above
  if (ly < HUD_MARGIN) ly = clamp(y + 4, HUD_MARGIN, vh - th - HUD_MARGIN);
  if (lx + tw + 8 > vw - HUD_MARGIN) lx = vw - tw - 8 - HUD_MARGIN;
  if (lx < HUD_MARGIN) lx = HUD_MARGIN;
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(lx - 4, ly, tw + 8, th);
  ctx.fillStyle = glow;
  ctx.fillText(txt, lx, ly + th - 5);
  ctx.restore();
}

// ---- Model config ----
const MODEL_URLS = [
  './yolov8n.onnx',
  'https://raw.githubusercontent.com/keirduffy25/V-Scanner/main/yolov8n.onnx' // fallback
];
const NET_W = 640, NET_H = 640;
const CONF_THRESHOLD = 0.15;
const IOU_THRESHOLD  = 0.40;

let session = null;
let inputName = 'images';
let running = false;
let scaleInfo = { vw: 1280, vh: 720, r: 1, dx: 0, dy: 0 };

// ---- Camera (user gesture required on iPad Chrome) ----
async function startCamera() {
  return new Promise((resolve) => {
    async function enable() {
      btnGo.textContent = 'Starting…'; btnGo.disabled = true;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' } }, audio: false
        });
        video.srcObject = stream;
        await video.play();
        // show video layer
        Object.assign(video.style, { position:'fixed', inset:'0', width:'100vw', height:'100vh', objectFit:'cover', zIndex: 0 });
        scaleInfo.vw = video.videoWidth || window.innerWidth;
        scaleInfo.vh = video.videoHeight || window.innerHeight;
        btnGo.style.display = 'none';
        btnStop.style.display = 'inline-block';
        resolve(true);
      } catch (e) {
        console.error(e);
        setStatus('⚠️ Camera access denied or unavailable', 'Tap address/lock → Allow Camera');
        btnGo.textContent = 'Activate Scanner'; btnGo.disabled = false;
        resolve(false);
      }
    }
    btnGo.addEventListener('click', enable, { once: true });
  });
}

// ---- ONNX model load ----
async function loadModel() {
  setStatus('Checking model URL…');
  for (const url of MODEL_URLS) {
    try {
      const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      if (!head.ok) throw new Error(`HTTP ${head.status}`);
      setStatus('Loading YOLOv8 (.onnx)…');
      // Use WASM for widest compatibility (WebGL can be flaky on iPad)
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      inputName = session.inputNames?.[0] || 'images';
      setStatus('Model loaded ✓', `Model input: ${inputName}`);
      return;
    } catch (e) {
      console.warn('Model load failed from', url, e);
    }
  }
  throw new Error('Failed to load YOLO model (checked all URLs).');
}

// ---- Preprocess: video → 1x3x640x640 Float32 (CHW, 0..1) ----
const work = document.createElement('canvas');
work.width = NET_W; work.height = NET_H;
const wctx = work.getContext('2d', { willReadFrequently: true });

function preprocess() {
  const vw = video.videoWidth  || scaleInfo.vw;
  const vh = video.videoHeight || scaleInfo.vh;

  const r = Math.min(NET_W / vw, NET_H / vh);
  const nw = Math.round(vw * r);
  const nh = Math.round(vh * r);
  const dx = Math.floor((NET_W - nw) / 2);
  const dy = Math.floor((NET_H - nh) / 2);

  wctx.clearRect(0, 0, NET_W, NET_H);
  wctx.fillStyle = '#000'; wctx.fillRect(0,0,NET_W,NET_H);
  wctx.drawImage(video, 0, 0, vw, vh, dx, dy, nw, nh);

  const img = wctx.getImageData(0,0,NET_W,NET_H).data;
  const chw = new Float32Array(3 * NET_W * NET_H);
  let rI = 0, gI = NET_W*NET_H, bI = 2*NET_W*NET_H;
  for (let i=0; i<img.length; i+=4) {
    chw[rI++] = img[i]   / 255;
    chw[gI++] = img[i+1] / 255;
    chw[bI++] = img[i+2] / 255;
  }

  // save mapping info for reverse transform
  scaleInfo = { vw, vh, r, dx, dy };
  return new ort.Tensor('float32', chw, [1, 3, NET_H, NET_W]);
}

// ---- Simple NMS ----
function nms(boxes, scores, iouThr, limit=100) {
  const keep = [];
  const areas = boxes.map(b => (b[2]-b[0])*(b[3]-b[1]));
  const order = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  while (order.length && keep.length < limit) {
    const i = order.shift(); keep.push(i);
    const keepNext = [];
    for (const j of order) {
      const xx1 = Math.max(boxes[i][0], boxes[j][0]);
      const yy1 = Math.max(boxes[i][1], boxes[j][1]);
      const xx2 = Math.min(boxes[i][2], boxes[j][2]);
      const yy2 = Math.min(boxes[i][3], boxes[j][3]);
      const w = Math.max(0, xx2 - xx1), h = Math.max(0, yy2 - yy1);
      const inter = w * h;
      const ovr = inter / (areas[i] + areas[j] - inter);
      if (ovr <= iouThr) keepNext.push(j);
    }
    order.splice(0, order.length, ...keepNext);
  }
  return keep;
}

// ---- YOLOv8 decode (handles [1,84,N] and [1,N,84]/[1,25200,85]) ----
const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie',
'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
'cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
'clock','vase','scissors','teddy bear','hair drier','toothbrush'];

function decode(outputs) {
  const out = outputs[session.outputNames[0]];
  if (!out) return [];
  const d = out.dims;           // e.g. [1,84,N] or [1,N,84] or [1,25200,85]
  const data = out.data;

  const boxes=[]; const scores=[]; const classes=[];

  // A) [1, 84/85, N]
  if (d.length === 3 && d[1] >= 84) {
    const C = d[1], N = d[2];
    const clsStart = (C > 84) ? 5 : 4;
    const hasObj   = (C > 84);
    for (let i=0;i<N;i++){
      const x = data[0*N + i], y = data[1*N + i], w = data[2*N + i], h = data[3*N + i];
      const obj = hasObj ? data[4*N + i] : 1.0;
      let best = 0, bi = -1;
      for (let c=clsStart;c<C;c++){ const v = data[c*N + i]; if (v > best){ best = v; bi=c-clsStart; } }
      const score = best * obj;
      if (score < CONF_THRESHOLD || bi < 0) continue;

      // xywh@640 → xyxy in video space
      let x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2;
      x1 = (x1 - scaleInfo.dx) / scaleInfo.r; y1 = (y1 - scaleInfo.dy) / scaleInfo.r;
      x2 = (x2 - scaleInfo.dx) / scaleInfo.r; y2 = (y2 - scaleInfo.dy) / scaleInfo.r;
      boxes.push([x1,y1,x2,y2]); scores.push(score); classes.push(bi);
    }
  }
  // B) [1, N, 84/85]
  else if (d.length === 3 && (d[2] >= 84 || d[2] === 85)) {
    const N = d[1], C = d[2];
    const clsStart = (C > 84) ? 5 : 4;
    const hasObj   = (C > 84);
    for (let i=0;i<N;i++){
      const base = i*C;
      const x = data[base+0], y = data[base+1], w = data[base+2], h = data[base+3];
      const obj = hasObj ? data[base+4] : 1.0;
      let best = 0, bi = -1;
      for (let c=clsStart;c<C;c++){ const v = data[base+c]; if (v > best){ best=v; bi=c-clsStart; } }
      const score = best * obj;
      if (score < CONF_THRESHOLD || bi < 0) continue;

      let x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2;
      x1 = (x1 - scaleInfo.dx) / scaleInfo.r; y1 = (y1 - scaleInfo.dy) / scaleInfo.r;
      x2 = (x2 - scaleInfo.dx) / scaleInfo.r; y2 = (y2 - scaleInfo.dy) / scaleInfo.r;
      boxes.push([x1,y1,x2,y2]); scores.push(score); classes.push(bi);
    }
  } else {
    console.warn('Unknown output shape', d);
  }

  const keep = nms(boxes, scores, IOU_THRESHOLD, 50);
  return keep.map(i => {
    const [x1,y1,x2,y2] = boxes[i];
    const w = x2-x1, h = y2-y1;
    return { x:x1, y:y1, w, h, score: scores[i], name: COCO[classes[i]] || `c${classes[i]}` };
  });
}

// ---- Main loop ----
async function loop() {
  if (!running) return;
  const w = window.innerWidth, h = window.innerHeight;
  ctx.clearRect(0,0,w,h);

  // Draw HUD status
  drawHUD();

  if (video.readyState >= 2 && session) {
    const tensor = preprocess();
    const outputs = await session.run({ [inputName]: tensor });
    const dets = decode(outputs);

    if (dets.length === 0) {
      // visible fallback so you know drawing works
      drawDetBox(50,50,200,200,'scan',0.99,'#ff4d4d');
      setStatus('Scanner running ✓', 'Detections: 0');
    } else {
      dets.forEach(d => {
        // blue for living, green for furniture/objects
        const n = (d.name || '').toLowerCase();
        const glow = ['person','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'].includes(n)
          ? '#00b3ff' : '#00ff88';
        drawDetBox(d.x, d.y, d.w, d.h, d.name || 'obj', d.score, glow);
      });
      setStatus('Scanner running ✓', `Detections: ${dets.length}`);
    }
  }

  requestAnimationFrame(loop);
}

// ---- Start/Stop ----
async function start() {
  if (!window.ort || !ort.InferenceSession) {
    alert('onnxruntime-web failed to load before script.js.\nCheck index.html order and defer.');
    return;
  }
  setStatus('Tap Activate Scanner to begin…'); drawHUD();

  const camOk = await startCamera();
  if (!camOk) return;

  try { await loadModel(); }
  catch (e) { setStatus('❌ Model failed to load'); drawHUD(); return; }

  // resize once video has dimensions
  resizeCanvas();
  running = true;
  requestAnimationFrame(loop);
}

function stop() {
  running = false;
  ctx.clearRect(0,0,window.innerWidth,window.innerHeight);
  setStatus('Scanner stopped'); drawHUD();
  btnStop.style.display = 'none';
  btnGo.style.display = 'inline-block';
  btnGo.textContent = 'Activate Scanner'; btnGo.disabled = false;
}

// ---- Wire UI ----
window.addEventListener('load', () => {
  video.setAttribute('playsinline', '');
  setStatus('Tap Activate Scanner to begin…'); drawHUD();
  btnGo.addEventListener('click', start, { passive: true });
  btnStop.addEventListener('click', stop,  { passive: true });
});

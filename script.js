/* =========================
   Cyberpunk V-Scanner v1
   YOLOv8 (ONNX, onnxruntime-web)
   ========================= */

const CONF_THRES = 0.10;  // lower to see more detections
const NMS_IOU    = 0.50;
const TOP_K      = 300;   // soft cap before NMS

// Elements
const video   = document.getElementById('cam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const hud     = document.getElementById('hud') || makeHud();

// Globals
let session = null;
let running = false;
let modelInput = {w:640, h:640}; // YOLO default
let letterbox = {sx:0, sy:0, gain:1}; // for coordinate mapping
let rafId = null;

// ---- On load ----
boot();

async function boot() {
  writeHud(['Scanner running ✓']);
  await ensureORT();
  await ensureCamera();
  pingHud('Camera ready ✓');

  session = await loadModel();
  if (!session) {
    hudError('Failed to load YOLO model.');
    return;
  }
  writeHud(['Model loaded ✓','Tap Activate Scanner to begin…']);
  fitCanvas();
  window.addEventListener('resize', fitCanvas);
}

// -------------- UI helpers --------------
function makeHud(){
  const el = document.createElement('div');
  el.id = 'hud';
  el.style.cssText = `
    position:fixed;left:12px;top:12px;z-index:20;
    max-width:min(60vw,360px); font:12px/1.2 ui-monospace,monospace;
    color:#d8ffe6;background:rgba(0,20,20,.35);border:1px solid #2bd3a3;
    border-radius:8px;padding:8px 10px;backdrop-filter:blur(2px);
    text-shadow:0 0 6px #00ffc8;
  `;
  document.body.appendChild(el);
  return el;
}
function writeHud(lines){
  hud.innerHTML = lines.map(t=>`<div>• ${escapeHtml(t)}</div>`).join('');
}
function pingHud(line){
  hud.innerHTML += `<div>• ${escapeHtml(line)}</div>`;
}
function hudError(line){
  hud.innerHTML += `<div style="color:#ffbdbd">✖ ${escapeHtml(line)}</div>`;
}
function escapeHtml(s){return String(s).replace(/[&<>"]/g,m=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;' }[m]))}

// -------------- ORT + model --------------
async function ensureORT(){
  if (window.ort) return;
  // Load ORT web (wasm) if not included by HTML
  await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js');
}

async function loadModel(){
  // Prefer local model next to index.html
  const urls = [
    './yolov8n.onnx',
    'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx',
    'https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/ONNX/yolov8n.onnx'
  ];
  for (const url of urls){
    try {
      pingHud(`Loading YOLOv8 from: ${url.startsWith('.')?'(local)…':'(cdn)…'}`);
      const sess = await ort.InferenceSession.create(url, { executionProviders:['wasm'] });
      return sess;
    } catch(e){
      console.warn('Model load failed:', url, e);
    }
  }
  return null;
}

// -------------- Camera --------------
async function ensureCamera(){
  const st = { video: { facingMode: 'environment' }, audio: false };
  const stream = await navigator.mediaDevices.getUserMedia(st);
  video.srcObject = stream;
  await video.play();
}

function fitCanvas(){
  const w = video.videoWidth || window.innerWidth;
  const h = video.videoHeight || window.innerHeight;
  canvas.width  = w;
  canvas.height = h;

  // compute letterbox gain and offsets to map model->screen
  const gw = modelInput.w, gh = modelInput.h;
  const gain = Math.min(w / gw, h / gh);
  const nw = Math.round(gw * gain);
  const nh = Math.round(gh * gain);
  letterbox.gain = gain;
  letterbox.sx = Math.floor((w - nw) / 2);
  letterbox.sy = Math.floor((h - nh) / 2);
}

// -------------- Controls --------------
window.startScanner = async function startScanner(){
  if (!session) { hudError('Model not ready'); return; }
  running = true;
  loop();
}
window.stopScanner = function stopScanner(){
  running = false;
  cancelAnimationFrame(rafId);
  ctx.clearRect(0,0,canvas.width,canvas.height);
}

// -------------- Main loop --------------
async function loop(){
  if (!running) return;
  rafId = requestAnimationFrame(loop);

  // 1) draw video to hidden letterboxed area
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const w = canvas.width, h = canvas.height;
  const gw = modelInput.w, gh = modelInput.h;

  // draw video into letterbox rect (for preview we just overlay boxes; video element is visible under canvas)

  // 2) build tensor from current frame
  const imgData = frameToTensor(video, gw, gh); // {tensor, gain, sx, sy}
  // 3) run ONNX
  let out;
  try {
    out = await session.run({ images: imgData.tensor });
  } catch(e){
    console.warn('ONNX inference failed:', e);
    return;
  }
  const first = out[Object.keys(out)[0]];
  if (!first) return;

  // 4) decode predictions, NMS, draw
  const dets = decode(first, CONF_THRES).slice(0, TOP_K);
  const final = nms(dets, NMS_IOU);
  drawDetections(final);

  // cleanup tensor
  imgData.tensor.dispose?.();
}

// -------------- Tensor helpers --------------
function frameToTensor(videoEl, gw, gh){
  // read pixels via an offscreen canvas
  const off = frameToTensor._cv || (frameToTensor._cv = document.createElement('canvas'));
  const oc  = frameToTensor._ctx || (frameToTensor._ctx = off.getContext('2d', { willReadFrequently:true }));
  off.width = gw; off.height = gh;
  // draw with cover-fit from video to model size (same math as in fitCanvas but inverse)
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const gain = Math.max(gw / vw, gh / vh);
  const dw = Math.round(vw * gain), dh = Math.round(vh * gain);
  const dx = Math.floor((gw - dw) / 2), dy = Math.floor((gh - dh) / 2);
  oc.drawImage(videoEl, 0, 0, vw, vh, dx, dy, dw, dh);

  const data = oc.getImageData(0,0,gw,gh).data;
  // NHWC uint8 -> NCHW float32 normalized 0..1
  const size = gw * gh;
  const chw = new Float32Array(3 * size);
  for (let i=0, p=0; i<size; i++, p+=4){
    chw[i] = data[p]   / 255;          // R
    chw[i +   size] = data[p+1] / 255; // G
    chw[i + 2*size] = data[p+2] / 255; // B
  }
  const tensor = new ort.Tensor('float32', chw, [1,3,gh,gw]);
  return { tensor };
}

// -------------- Decode YOLOv8 --------------
function decode(tensor, confThres){
  // Accept output in either [1,8400,84] or [1,84,8400]
  const dims = tensor.dims;          // e.g., [1,8400,84] or [1,84,8400]
  const data = tensor.data;          // Float32Array
  const strideBoxesFirst = (dims[1] > dims[2]); // true if 8400>84 => [8400,84]

  const num = strideBoxesFirst ? dims[1] : dims[2];
  const classes = strideBoxesFirst ? dims[2]-4 : dims[1]-4;

  const dets = [];
  if (strideBoxesFirst) {
    // [1, 8400, 4+cls]
    for (let i=0; i<num; i++){
      const off = i * (4 + classes);
      const bx = data[off+0], by = data[off+1], bw = data[off+2], bh = data[off+3];
      // best class conf
      let bestC = -1, bestP = 0;
      for (let c=0; c<classes; c++){
        const p = data[off+4+c];
        if (p > bestP){ bestP = p; bestC = c; }
      }
      if (bestP < confThres) continue;
      dets.push({ cx:bx, cy:by, w:bw, h:bh, conf:bestP, cls:bestC });
    }
  } else {
    // [1, 4+cls, 8400] => split channels
    const stride = dims[2];
    const bx = data.subarray(0*stride, 1*stride);
    const by = data.subarray(1*stride, 2*stride);
    const bw = data.subarray(2*stride, 3*stride);
    const bh = data.subarray(3*stride, 4*stride);
    // classes start at channel 4
    for (let i=0; i<stride; i++){
      let bestC = -1, bestP = 0;
      for (let c=0; c<classes; c++){
        const p = data[(4+c)*stride + i];
        if (p > bestP){ bestP = p; bestC = c; }
      }
      if (bestP < confThres) continue;
      dets.push({ cx:bx[i], cy:by[i], w:bw[i], h:bh[i], conf:bestP, cls:bestC });
    }
  }
  return dets;
}

// -------------- NMS --------------
function nms(dets, iouThres){
  dets.sort((a,b)=>b.conf - a.conf);
  const res = [];
  const used = new Uint8Array(dets.length);
  for (let i=0;i<dets.length;i++){
    if (used[i]) continue;
    const a = dets[i];
    res.push(a);
    const ax1=a.cx-a.w/2, ay1=a.cy-a.h/2, ax2=a.cx+a.w/2, ay2=a.cy+a.h/2;
    for (let j=i+1;j<dets.length;j++){
      if (used[j]) continue;
      const b = dets[j];
      const bx1=b.cx-b.w/2, by1=b.cy-b.h/2, bx2=b.cx+b.w/2, by2=b.cy+b.h/2;
      const inter = Math.max(0, Math.min(ax2,bx2)-Math.max(ax1,bx1)) *
                    Math.max(0, Math.min(ay2,by2)-Math.max(ay1,by1));
      const ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
      const iou = ua>0 ? inter/ua : 0;
      if (iou > iouThres) used[j]=1;
    }
  }
  return res;
}

// -------------- Drawing --------------
function drawDetections(dets){
  const w = canvas.width, h = canvas.height;
  // Map model coords (640x640) to screen via letterbox
  const gain = Math.min(w / modelInput.w, h / modelInput.h);
  const xoff = (w - modelInput.w * gain) / 2;
  const yoff = (h - modelInput.h * gain) / 2;

  ctx.lineWidth = 2;
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';

  // HUD: show count
  writeHud([
    'Scanner running ✓',
    'Camera ready ✓',
    'Model loaded ✓',
    `Detections: ${dets.length}`
  ]);

  for (const d of dets){
    const x = (d.cx - d.w/2) * gain + xoff;
    const y = (d.cy - d.h/2) * gain + yoff;
    const ww = d.w * gain;
    const hh = d.h * gain;

    // box
    ctx.strokeStyle = 'rgba(0,255,200,0.9)';
    ctx.shadowColor = '#00ffc8';
    ctx.shadowBlur = 8;
    ctx.strokeRect(x,y,ww,hh);

    // label
    const label = `${className(d.cls)} ${(d.conf*100|0)}%`;
    const tw = ctx.measureText(label).width + 10;
    const th = 18;
    ctx.fillStyle = 'rgba(0,30,30,.75)';
    ctx.fillRect(x, y- th - 4, tw, th);
    ctx.strokeStyle = 'rgba(0,255,200,0.6)';
    ctx.strokeRect(x, y- th - 4, tw, th);
    ctx.fillStyle = '#d8ffe6';
    ctx.fillText(label, x+5, y- th - 2);
    ctx.shadowBlur = 0;
  }
}

function className(i){
  // COCO 80 classes (short list for label; you can replace with full array if you like)
  const short = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];
  return short[i] ?? `cls${i}`;
}

// -------------- tiny loader --------------
function loadScript(src){
  return new Promise((res,rej)=>{
    const s=document.createElement('script');
    s.src=src; s.onload=res; s.onerror=rej; document.head.appendChild(s);
  });
}

/* Cyberpunk V-Scanner (YOLOv8 ONNX, no yolo.min.js)
   - Designed for iPad/iPhone + GitHub Pages
   - Local yolov8n.onnx (repo root) preferred, CDN fallbacks
   - Reactive scan beam: faster/brighter when detecting
*/

const els = {
  video: document.getElementById('cam'),
  canvas: document.getElementById('overlay'),
  beam:   document.querySelector('.scan-beam'),
  toggle: document.getElementById('toggle'),
  hudState: document.getElementById('hud-state'),
  hudCam:   document.getElementById('hud-cam'),
  hudModel: document.getElementById('hud-model'),
  hudCount: document.getElementById('hud-count'),
};

const ctx = els.canvas.getContext('2d');

const MODEL_SIZE = 640;
const CONF_THRES = 0.28;
const NMS_IOU = 0.45;

// Try local ONNX first (works on GitHub Pages when yolov8n.onnx is in repo root)
const MODEL_URLS = [
  './yolov8n.onnx',
  // fallbacks (raw GitHub mirrors)
  'https://raw.githubusercontent.com/ultralytics/assets/main/onnx/models/yolov8n.onnx',
  'https://huggingface.co/onnx/models/resolve/main/vision/object-detection/yolov8/yolov8n.onnx'
];

let session = null;
let running = false;
let rafId = null;

// --- Camera init ---
async function initCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
      audio: false
    });
    els.video.srcObject = stream;

    // Wait for dimensions
    await new Promise(res=>{
      els.video.onloadedmetadata = ()=>res();
    });

    // Fit canvas to displayed video size
    resizeCanvasToVideo();
    window.addEventListener('resize', resizeCanvasToVideo, { passive:true });

    els.hudCam.textContent = 'ready ✓';
    return true;
  }catch(e){
    els.hudCam.textContent = 'permission needed';
    alert('Camera permission is required.');
    return false;
  }
}

function resizeCanvasToVideo(){
  const w = els.video.clientWidth || window.innerWidth;
  const h = els.video.clientHeight || window.innerHeight;
  els.canvas.width  = w;
  els.canvas.height = h;
}

// --- Load ONNX model with fallbacks ---
async function loadModel(){
  els.hudModel.textContent = 'loading…';
  let lastErr = null;
  for (const url of MODEL_URLS){
    try{
      // Prefer WASM; WebGL can be unstable on some mobile GPUs
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      els.hudModel.textContent = `loaded ✓`;
      return true;
    }catch(e){
      lastErr = e;
    }
  }
  console.warn('Model load failed:', lastErr);
  els.hudModel.textContent = 'failed ✗';
  alert('Failed to load YOLOv8 model. Ensure yolov8n.onnx sits next to index.html.');
  return false;
}

// --- Letterbox: pad to square MODEL_SIZE while keeping aspect ---
function letterbox(video, dst){
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const scale = Math.min(MODEL_SIZE/vw, MODEL_SIZE/vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const dx = Math.floor((MODEL_SIZE - nw) / 2);
  const dy = Math.floor((MODEL_SIZE - nh) / 2);

  dst.clearRect(0,0,MODEL_SIZE,MODEL_SIZE);
  dst.fillStyle = 'black';
  dst.fillRect(0,0,MODEL_SIZE,MODEL_SIZE);
  dst.drawImage(video, 0,0,vw,vh, dx,dy, nw,nh);

  return { scale, dx, dy, srcW: vw, srcH: vh, newW:nw, newH:nh };
}

// Convert ImageData to Float32 NCHW normalized
function toNCHWFloat(buffer){
  const { data, width, height } = buffer;
  const out = new Float32Array(3 * width * height);
  let p = 0, c0 = 0, c1 = width*height, c2 = 2*width*height;
  for(let i=0;i<data.length;i+=4){
    const r = data[i]/255, g = data[i+1]/255, b = data[i+2]/255;
    out[c0++] = r; out[c1++] = g; out[c2++] = b;
    p += 4;
  }
  return out;
}

// Simple NMS
function nms(boxes, scores, iou=0.45, max=100){
  const idx = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const selected = [];
  while(idx.length && selected.length<max){
    const i = idx.shift();
    selected.push(i);
    const rest = [];
    for(const j of idx){
      const iouVal = IoU(boxes[i], boxes[j]);
      if(iouVal < iou) rest.push(j);
    }
    idx.splice(0, idx.length, ...rest);
  }
  return selected;
}

function IoU(a,b){
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const w = Math.max(0, x2-x1);
  const h = Math.max(0, y2-y1);
  const inter = w*h;
  const areaA = (a[2]-a[0])*(a[3]-a[1]);
  const areaB = (b[2]-b[0])*(b[3]-b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

// Decode YOLOv8 (exported ONNX): supports [1,8400,84] or [1,84,8400]
function decode(output, imgMeta){
  // outputTensor.data is Float32Array
  const data = output.data;
  const dims = output.dims; // e.g. [1,8400,84] or [1,84,8400]
  let boxes=[], scores=[], classes=[];

  const strideBoxesFirst = (dims[1]===8400 && dims[2]===84);
  const num = strideBoxesFirst ? dims[1] : dims[2];

  for(let i=0;i<num;i++){
    // read row i
    let x,y,w,h, best=-1, conf=-Infinity;
    if(strideBoxesFirst){
      const base = i*84;
      x=data[base+0]; y=data[base+1]; w=data[base+2]; h=data[base+3];
      for(let c=4;c<84;c++){ const s=data[base+c]; if(s>conf){conf=s; best=c-4;} }
    }else{
      // [1,84,8400]
      const base = i;
      x=data[0*num+base]; y=data[1*num+base]; w=data[2*num+base]; h=data[3*num+base];
      for(let c=4;c<84;c++){ const s=data[c*num+base]; if(s>conf){conf=s; best=c-4;} }
    }
    if(conf < CONF_THRES) continue;

    // xywh -> xyxy in model space
    const x1 = x - w/2, y1 = y - h/2, x2 = x + w/2, y2 = y + h/2;

    // map back to canvas coords (undo letterbox)
    const { scale, dx, dy, srcW, srcH, newW, newH } = imgMeta;
    const rx = (val)=> (val - dx) / (newW/MODEL_SIZE) * (els.canvas.width/MODEL_SIZE);
    const ry = (val)=> (val - dy) / (newH/MODEL_SIZE) * (els.canvas.height/MODEL_SIZE);

    const bx1 = Math.max(0, rx(x1));
    const by1 = Math.max(0, ry(y1));
    const bx2 = Math.min(els.canvas.width,  rx(x2));
    const by2 = Math.min(els.canvas.height, ry(y2));

    if (bx2>bx1 && by2>by1){
      boxes.push([bx1,by1,bx2,by2]);
      scores.push(conf);
      classes.push(best);
    }
  }

  const keep = nms(boxes, scores, NMS_IOU, 100);
  return keep.map(i=>({ box:boxes[i], score:scores[i], cls:classes[i] }));
}

// Draw boxes
function draw(dets){
  ctx.clearRect(0,0,els.canvas.width, els.canvas.height);
  ctx.lineWidth = 2;
  for(const d of dets){
    const [x1,y1,x2,y2] = d.box;
    // box
    ctx.strokeStyle = 'rgba(0,255,213,0.9)';
    ctx.shadowColor = 'rgba(0,255,213,0.6)';
    ctx.shadowBlur = 6;
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.shadowBlur = 0;

    // label
    const label = `${d.cls} ${(d.score*100|0)}%`;
    const tw = ctx.measureText(label).width + 12;
    const th = 18;
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.strokeStyle = 'rgba(0,255,213,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(x1, Math.max(0,y1-22), tw, th, 6);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#eaffff';
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.fillText(label, x1+6, Math.max(0,y1-22)+13);
  }
}

// Reactive beam controller
function updateBeam(hasDetections){
  if(!els.beam) return;
  if(hasDetections){
    els.beam.classList.add('fast');
    els.beam.classList.remove('slow');
  }else{
    els.beam.classList.add('slow');
    els.beam.classList.remove('fast');
  }
}

// --- Main loop ---
const scratch = document.createElement('canvas');
scratch.width = scratch.height = MODEL_SIZE;
const sctx = scratch.getContext('2d', { willReadFrequently:true });

async function loop(){
  if(!running){ return; }

  // Prepare model input
  const meta = letterbox(els.video, sctx);
  const imgData = sctx.getImageData(0,0,MODEL_SIZE,MODEL_SIZE);
  const input = toNCHWFloat(imgData);
  const tensor = new ort.Tensor('float32', input, [1,3,MODEL_SIZE,MODEL_SIZE]);

  const results = await session.run({images: tensor}).catch(()=>null);
  if(!results){ requestNext(); return; }

  // pick first output
  const out = results[Object.keys(results)[0]];
  const dets = decode(out, meta);

  els.hudCount.textContent = dets.length;
  updateBeam(dets.length>0);
  draw(dets);

  requestNext();
}

function requestNext(){
  rafId = requestAnimationFrame(loop);
}

// --- UI ---
els.toggle.addEventListener('click', async ()=>{
  if(running){
    running = false;
    cancelAnimationFrame(rafId);
    els.toggle.textContent = 'Start';
    els.hudState.textContent = 'idle';
    updateBeam(false);
    ctx.clearRect(0,0,els.canvas.width, els.canvas.height);
    return;
  }

  els.toggle.disabled = true;
  els.hudState.textContent = 'starting…';

  const camOk = await initCamera();
  if(!camOk){ els.toggle.disabled=false; return; }

  if(!session){
    const ok = await loadModel();
    if(!ok){ els.toggle.disabled=false; return; }
  }

  running = true;
  els.toggle.textContent = 'Stop';
  els.hudState.textContent = 'running ✓';
  els.toggle.disabled = false;
  requestNext();
});

// Auto-start if permission already granted
(async ()=>{
  if(navigator.permissions){
    try{
      const st = await navigator.permissions.query({ name:'camera' });
      if(st.state==='granted'){
        els.toggle.click();
      }
    }catch{}
  }
})();

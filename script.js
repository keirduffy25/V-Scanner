// ===== V-Scanner (ONNX Runtime Web; debug-strong) =====

// ---- HUD helper -------------------------------------------------
const hud = (() => {
  const el = document.getElementById('hud');
  const lines = [];
  function push(msg, cls='') {
    lines.unshift({ msg, cls });
    if (lines.length > 12) lines.pop();
    el.innerHTML = lines.map(l => `<div class="${l.cls||''}">• ${l.msg}</div>`).join('');
  }
  return { push };
})();

// ---- elements ---------------------------------------------------
const video   = document.getElementById('cam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const startBtn= document.getElementById('startBtn');

// Make the video visible (as a quick sanity check)
video.style.position = 'fixed';
video.style.inset = '0';
video.style.width = '100%';
video.style.height = '100%';
video.style.objectFit = 'cover';
video.style.zIndex = '0';       // video underlay
canvas.style.zIndex = '1';      // HUD on top

// Model config
const MODEL_URL = './yolov8n.onnx';  // file must be in repo root
const INPUT_W = 640, INPUT_H = 640;
const CONF_THRESHOLD = 0.35, IOU_THRESHOLD = 0.45;

// COCO-80 labels
const LABELS = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie',
'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
'cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
'clock','vase','scissors','teddy bear','hair drier','toothbrush'];

// ---- safety: ensure ORT loaded ---------------------------------
document.addEventListener('DOMContentLoaded', () => {
  if (!window.ort || !ort.InferenceSession) {
    alert('onnxruntime-web failed to load before script.js.\nCheck index.html: ORT <script> must appear before script.js and both must use defer.');
    hud.push('Error: onnxruntime-web not loaded before script.js', 'err');
  } else {
    hud.push('ORT ready ✓');
  }
});

// ---- camera with user-gesture start ----------------------------
function resizeCanvasToVideo() {
  const w = video.videoWidth || window.innerWidth;
  const h = video.videoHeight || window.innerHeight;
  canvas.width = w; canvas.height = h;
}

async function startCameraAfterTap() {
  return new Promise((resolve) => {
    async function enableCamera() {
      startBtn.textContent = 'Starting…';
      startBtn.disabled = true;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' } },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
        resizeCanvasToVideo();
        window.addEventListener('resize', resizeCanvasToVideo);
        startBtn.style.display = 'none';
        hud.push('Camera ready ✓');
        resolve(true);
      } catch (e) {
        console.error(e);
        hud.push('Camera access denied or unavailable', 'warn');
        startBtn.textContent = 'Activate Scanner';
        startBtn.disabled = false;
        resolve(false);
      }
    }
    // ensure handler is attached (even if user taps quickly)
    startBtn.addEventListener('click', enableCamera, { once: true });
  });
}

// ---- ONNX helpers ----------------------------------------------
let session = null;
let inputName = 'images'; // default, will adjust to actual

async function loadModel() {
  hud.push('Checking model URL…');
  let head;
  try {
    head = await fetch(MODEL_URL, { method: 'HEAD', cache: 'no-store' });
  } catch (e) {
    hud.push('Network error reaching yolov8n.onnx', 'err');
    alert('Network error fetching yolov8n.onnx. Check your connection.');
    throw e;
  }
  if (!head.ok) {
    hud.push(`Model not reachable (HTTP ${head.status})`, 'err');
    alert(`Failed to load YOLO model (HTTP ${head.status}).\nEnsure yolov8n.onnx is in the repo root.`);
    throw new Error('Model HEAD failed');
  }
  hud.push('Loading YOLOv8 (.onnx)…');

  // WASM for widest compatibility on iPad Chrome
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });
  hud.push('Model loaded ✓');

  // Get real input name (could be 'images' or 'input')
  inputName = session.inputNames?.[0] || 'images';
  hud.push(`Model input: ${inputName}`);
}

// letterbox + CHW float32 [1,3,640,640]
function prepareInputFromVideo() {
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;

  const scale = Math.min(INPUT_W / vw, INPUT_H / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padW = Math.floor((INPUT_W - nw) / 2);
  const padH = Math.floor((INPUT_H - nh) / 2);

  const tmp = document.createElement('canvas');
  tmp.width = INPUT_W; tmp.height = INPUT_H;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = '#000'; tctx.fillRect(0,0,INPUT_W,INPUT_H);
  tctx.drawImage(video, 0, 0, vw, vh, padW, padH, nw, nh);

  const img = tctx.getImageData(0,0,INPUT_W,INPUT_H).data;
  const chw = new Float32Array(3 * INPUT_W * INPUT_H);
  let p = 0;
  for (let i=0;i<img.length;i+=4) {
    const r = img[i]   / 255;
    const g = img[i+1] / 255;
    const b = img[i+2] / 255;
    chw[p] = r;
    chw[p + INPUT_W*INPUT_H] = g;
    chw[p + 2*INPUT_W*INPUT_H] = b;
    p++;
  }
  return { chw, scale, padW, padH, vw, vh };
}

// simple NMS
function nms(boxes, scores, iouThresh, limit=100) {
  const picked = [];
  const areas = boxes.map(b => (b[2]-b[0])*(b[3]-b[1]));
  const order = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);

  while(order.length && picked.length<limit){
    const i = order.shift();
    picked.push(i);
    const keep=[];
    for(const j of order){
      const xx1=Math.max(boxes[i][0],boxes[j][0]);
      const yy1=Math.max(boxes[i][1],boxes[j][1]);
      const xx2=Math.min(boxes[i][2],boxes[j][2]);
      const yy2=Math.min(boxes[i][3],boxes[j][3]);
      const w=Math.max(0,xx2-xx1), h=Math.max(0,yy2-yy1);
      const inter=w*h;
      const ovr=inter/(areas[i]+areas[j]-inter);
      if(ovr<=iouThresh) keep.push(j);
    }
    order.splice(0,order.length,...keep);
  }
  return picked;
}

// decode Ultralytics head: [1,84,N] or [1,N,84]
function decode(outputs, scaleInfo) {
  const { scale, padW, padH, vw, vh } = scaleInfo;
  const out = Object.values(outputs)[0];
  if (!out) return [];
  let data = out.data, dims = out.dims;

  if (dims[1] !== 84 && dims[2] === 84) {
    // transpose [1,N,84] -> [1,84,N]
    const N = dims[1], tmp = new Float32Array(84*N);
    for (let n=0;n<N;n++) for (let c=0;c<84;c++) tmp[c*N+n] = data[n*84 + c];
    data = tmp; dims = [1,84,N];
  }
  if (dims[1] !== 84) return [];

  const N = dims[2];
  const boxes=[], scores=[], classes=[];
  for (let i=0;i<N;i++){
    const x = data[0*N+i], y = data[1*N+i], w = data[2*N+i], h = data[3*N+i];
    let best=0, cls=0;
    for (let c=4;c<84;c++){
      const v = data[c*N+i]; if (v>best){best=v; cls=c-4;}
    }
    if (best < CONF_THRESHOLD) continue;

    const cx=(x - padW)/scale, cy=(y - padH)/scale, ww=w/scale, hh=h/scale;
    const x1=Math.max(0,cx-ww/2), y1=Math.max(0,cy-hh/2);
    const x2=Math.min(vw,cx+ww/2), y2=Math.min(vh,cy+hh/2);
    boxes.push([x1,y1,x2,y2]); scores.push(best); classes.push(cls);
  }
  const keep = nms(boxes, scores, IOU_THRESHOLD, 50);
  return keep.map(i => ({ box: boxes[i], score: scores[i], cls: classes[i] }));
}

function draw(results) {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  for (const r of results) {
    const [x1,y1,x2,y2]=r.box, w=x2-x1, h=y2-y1;
    const name = LABELS[r.cls] || 'obj';
    const label = `${name} ${(r.score*100|0)}%`;

    let glow = '#00e5ff';
    const lower = name.toLowerCase();
    if (['person','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'].includes(lower)) glow = '#00b3ff';
    else if (['chair','couch','bench','bed','toilet','tv','laptop','keyboard','cell phone','book','clock','vase'].includes(lower)) glow = '#00ff88';

    ctx.lineWidth = 2;
    ctx.strokeStyle = glow;
    ctx.shadowColor = glow;
    ctx.shadowBlur = 12;
    ctx.strokeRect(x1,y1,w,h);

    ctx.shadowBlur = 0;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.font = '12px ui-monospace,monospace';
    const tw = ctx.measureText(label).width + 10;
    ctx.fillRect(x1, Math.max(0, y1-18), tw, 18);
    ctx.fillStyle = glow;
    ctx.fillText(label, x1+5, y1-5);
  }
}

// ---- main -------------------------------------------------------
(async function main(){
  try {
    hud.push('Tap Activate Scanner to begin…');

    const camOk = await startCameraAfterTap();
    if (!camOk) return;

    try {
      await loadModel();
    } catch (e) {
      return; // error already shown
    }

    // detection loop
    async function loop() {
      const { chw, ...scaleInfo } = prepareInputFromVideo();
      const tensor = new ort.Tensor('float32', chw, [1,3,INPUT_H,INPUT_W]);
      const outputs = await session.run({ [inputName]: tensor });
      const dets = decode(outputs, scaleInfo);
      draw(dets);
      requestAnimationFrame(loop);
    }
    loop();
    hud.push('Scanner running ✓');

  } catch (err) {
    console.error(err);
    hud.push(`Fatal: ${err.message||err}`, 'err');
  }
})();

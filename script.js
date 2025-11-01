/* =========================
   Cyberpunk V-Scanner (iPad)
   ONNX Runtime Web + YOLOv8n.onnx
   ========================= */

const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d', { alpha: true });

const btn = document.getElementById('scanBtn'); // optional button (#scanBtn)
const hud = document.getElementById('hud') || makeHud(); // create if not in HTML

// State
let scanning = false;
let rafId = null;
let session = null;
let modelLoaded = false;
let inputShape = [1, 3, 640, 640]; // default YOLOv8n
let letterboxMeta = null;

// ONNX Runtime preferred backends (GPU first if available)
const ortOptions = {
  executionProviders: ['webgpu', 'wasm'],
  // improves perf on iPad/Chrome
  graphOptimizationLevel: 'all'
};

// ---------- HUD helpers ----------
function makeHud() {
  const el = document.createElement('div');
  el.id = 'hud';
  el.setAttribute('role', 'status');
  el.style.cssText = 'position:fixed;left:12px;top:12px;z-index:10000;';
  document.body.appendChild(el);
  injectHudStyles();
  return el;
}

function injectHudStyles() {
  if (document.getElementById('hud-style')) return;
  const css = `
  #hud {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 12px;
    line-height: 1.35;
    color: #bdfcff;
    max-width: min(42ch, 92vw);
    padding: 8px 10px;
    border: 1px solid rgba(0,255,255,.35);
    border-radius: 8px;
    background: rgba(0,10,20,.55);
    box-shadow: 0 0 0 1px rgba(0,255,255,.15), 0 8px 24px rgba(0,0,0,.35);
    pointer-events: none;
    transition: opacity .4s ease;
  }
  #hud.fade { opacity: .15; }
  #hud .ok::before   { content: "• "; color: #7dffbf; }
  #hud .warn::before { content: "• "; color: #ffd36f; }
  #hud .err::before  { content: "• "; color: #ff7d7d; }
  .chip {
    position: fixed;
    left: 50%; bottom: 18px; transform: translateX(-50%);
    padding: 8px 14px; border-radius: 10px;
    background: rgba(0,10,20,.55);
    color: #bdfcff; border: 1px solid rgba(0,255,255,.35);
    box-shadow: 0 0 0 1px rgba(0,255,255,.15), 0 8px 24px rgba(0,0,0,.35);
    font: 600 14px/1 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    z-index: 10001;
  }
  canvas#overlay {
    position: absolute; inset: 0; z-index: 9000;
    pointer-events: none;
  }
  `;
  const tag = document.createElement('style');
  tag.id = 'hud-style'; tag.textContent = css;
  document.head.appendChild(tag);
}

let hudFadeTimer = null;
function hudReset() {
  hud.innerHTML = '';
  hud.classList.remove('fade');
  clearTimeout(hudFadeTimer);
  hudFadeTimer = setTimeout(() => hud.classList.add('fade'), 3500);
}
function hudLine(text, cls='ok') {
  const p = document.createElement('div');
  p.className = cls;
  p.textContent = text;
  hud.appendChild(p);
}

// Wake HUD on interaction
['click','touchstart','mousemove'].forEach(ev => {
  window.addEventListener(ev, () => {
    hud.classList.remove('fade');
    clearTimeout(hudFadeTimer);
    hudFadeTimer = setTimeout(() => hud.classList.add('fade'), 2500);
  }, { passive: true });
});

// ---------- Camera / Canvas ----------
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' }, audio: false
    });
    video.srcObject = stream;
    await video.play();
    hudLine('Camera ready ✓', 'ok');
  } catch (e) {
    hudLine('Camera error: ' + e.message, 'err');
    throw e;
  }
}

function resizeOverlayToVideo() {
  const { videoWidth: w, videoHeight: h } = video;
  if (!w || !h) return;
  overlay.width = w;
  overlay.height = h;
  // Ensure overlay sits over the video element even if page scrolls
  const rect = video.getBoundingClientRect();
  overlay.style.left = rect.left + window.scrollX + 'px';
  overlay.style.top  = rect.top + window.scrollY + 'px';
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
}

// For iPad orientation/layout changes
function updateViewRect() {
  resizeOverlayToVideo();
}
window.addEventListener('resize', updateViewRect);
window.addEventListener('orientationchange', () => setTimeout(updateViewRect, 300));

// ---------- Model loading (once) ----------
async function loadModel() {
  // Already have a session? Bail.
  if (session) return;

  const local = './yolov8n.onnx'; // your GitHub Pages local file
  const cdn1 = 'https://huggingface.co/onnx/models/resolve/main/vision/object-detection/yolov8/yolov8n.onnx';
  const cdn2 = 'https://raw.githubusercontent.com/ultralytics/assets/main/onnx/yolov8n.onnx';

  const urls = [local, cdn1, cdn2];

  let lastErr = null;
  for (const url of urls) {
    try {
      hudLine(`Loading YOLOv8 from: ${labelUrl(url)}…`, 'warn');
      session = await ort.InferenceSession.create(url, ortOptions);
      hudLine('Model loaded ✓', 'ok');
      // Optional warmup for smoother first frame:
      await warmup();
      modelLoaded = true;
      return;
    } catch (e) {
      lastErr = e;
    }
  }
  hudLine('Failed to load model: ' + (lastErr?.message || lastErr), 'err');
  throw lastErr;
}

function labelUrl(url) {
  if (url.startsWith('./')) return '(local)';
  if (url.includes('huggingface')) return '(cdn:hf)';
  if (url.includes('raw.githubusercontent')) return '(cdn:gh)';
  return '(url)';
}

async function warmup() {
  // Make a black input of model shape
  const [n, c, ih, iw] = inputShape;
  const data = new Float32Array(n * c * ih * iw);
  const input = new ort.Tensor('float32', data, [n, c, ih, iw]);
  const feeds = { images: input };
  try { await session.run(feeds); } catch(_) {}
}

// ---------- Pre/Post processing ----------
function toLetterboxedTensor(img, size=640) {
  // Resize + pad to square while preserving aspect ratio (letterbox like YOLO)
  const iw = img.videoWidth || img.width;
  const ih = img.videoHeight || img.height;

  const scale = Math.min(size/iw, size/ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);
  const padW = size - nw;
  const padH = size - nh;
  const padLeft = Math.floor(padW/2);
  const padTop  = Math.floor(padH/2);

  // Draw to temp canvas
  const tmp = document.createElement('canvas');
  tmp.width = size; tmp.height = size;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = 'black';
  tctx.fillRect(0,0,size,size);
  tctx.drawImage(img, 0,0, iw, ih, padLeft, padTop, nw, nh);

  // Get RGB float
  const imgData = tctx.getImageData(0,0,size,size).data;
  const float = new Float32Array(3 * size * size);
  // Normalize 0..1 and CHW
  let p = 0;
  for (let y=0; y<size; y++) {
    for (let x=0; x<size; x++) {
      const i = (y*size + x)*4;
      const r = imgData[i]   / 255;
      const g = imgData[i+1] / 255;
      const b = imgData[i+2] / 255;
      float[p] = r;                     // R
      float[p + size*size] = g;        // G
      float[p + 2*size*size] = b;      // B
      p++;
    }
  }

  letterboxMeta = { scale, padLeft, padTop, inW: iw, inH: ih, size };
  return new ort.Tensor('float32', float, [1,3,size,size]);
}

function yolov8Postprocess(output, imgW, imgH) {
  // YOLOv8 ONNX gives shape [1, 84, N]; 4 box + 80 classes
  const out = output.output0?.data || output[Object.keys(output)[0]].data;
  const rows = out.length / 84; // number of boxes
  const size = letterboxMeta.size;
  const scale = 1/letterboxMeta.scale;
  const padL = letterboxMeta.padLeft;
  const padT = letterboxMeta.padTop;

  const results = [];
  for (let i=0;i<rows;i++){
    const off = i*84;
    const cx = out[off+0], cy = out[off+1], w = out[off+2], h = out[off+3];
    // best class
    let bestScore = 0, bestIdx = -1;
    for (let c=4;c<84;c++){
      const sc = out[off+c];
      if (sc > bestScore) { bestScore = sc; bestIdx = c-4; }
    }
    if (bestScore < 0.35) continue; // threshold

    // Convert center xywh -> xyxy in letterbox space
    let x1 = cx - w/2;
    let y1 = cy - h/2;
    let x2 = cx + w/2;
    let y2 = cy + h/2;

    // Undo letterbox to original video pixels
    x1 = Math.max(0, Math.min(imgW, (x1 - padL) * scale));
    y1 = Math.max(0, Math.min(imgH, (y1 - padT) * scale));
    x2 = Math.max(0, Math.min(imgW, (x2 - padL) * scale));
    y2 = Math.max(0, Math.min(imgH, (y2 - padT) * scale));

    results.push({ x1, y1, x2, y2, cls: bestIdx, conf: bestScore });
  }

  // Simple NMS-lite by confidence sort
  results.sort((a,b)=>b.conf-a.conf);
  return results.slice(0, 20);
}

// ---------- Drawing ----------
function draw(results) {
  const W = overlay.width, H = overlay.height;
  ctx.clearRect(0,0,W,H);

  // Pick the highest-confidence target to “lock” center box
  const focus = results[0];

  // Draw all boxes
  for (const det of results) {
    const { x1,y1,x2,y2, conf } = det;
    // Clamp inside canvas with 2px padding
    const pad = 2;
    const rx1 = Math.max(pad, Math.min(W-pad, x1));
    const ry1 = Math.max(pad, Math.min(H-pad, y1));
    const rw  = Math.max(0, Math.min(W-2*pad, x2 - rx1));
    const rh  = Math.max(0, Math.min(H-2*pad, y2 - ry1));

    ctx.lineWidth = (det === focus) ? 3 : 2;
    ctx.strokeStyle = (det === focus) ? '#00ffd5' : 'rgba(0,255,213,.6)';
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 8;

    ctx.strokeRect(rx1, ry1, rw, rh);

    // Label
    const label = `${Math.round(conf*100)}%`;
    const tw = ctx.measureText(label).width + 12;
    const th = 18;

    ctx.fillStyle = 'rgba(0,10,20,.65)';
    ctx.strokeStyle = 'rgba(0,255,213,.6)';
    ctx.lineWidth = 1.5;
    ctx.fillRect(rx1, Math.max(0, ry1 - th - 6), tw, th);
    ctx.strokeRect(rx1, Math.max(0, ry1 - th - 6), tw, th);

    ctx.fillStyle = '#bdfcff';
    ctx.font = '600 12px ui-sans-serif, system-ui, -apple-system';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, rx1 + 6, Math.max(9, ry1 - th/2 - 6));
  }
}

// ---------- Main loop ----------
async function tick() {
  if (!scanning) return;
  try {
    const input = toLetterboxedTensor(video, 640);
    const feeds = { images: input };
    const out = await session.run(feeds);
    const results = yolov8Postprocess(out, video.videoWidth, video.videoHeight);
    draw(results);
  } catch (e) {
    hudLine('Inference error: ' + e.message, 'err');
  }
  rafId = requestAnimationFrame(tick);
}

// ---------- Controls ----------
async function startScanner() {
  if (scanning) return;

  hudReset();
  hudLine('Scanner running ✓');

  await initCamera();
  resizeOverlayToVideo();
  updateViewRect();

  if (!session && !modelLoaded) {
    await loadModel();
  } else {
    hudLine('Model already loaded ✓');
  }

  scanning = true;
  toggleChip(true);
  rafId = requestAnimationFrame(tick);
}

function stopScanner() {
  scanning = false;
  if (rafId) cancelAnimationFrame(rafId);
  ctx.clearRect(0,0,overlay.width, overlay.height);
  toggleChip(false);
}

function toggleChip(on) {
  let chip = document.getElementById('chip');
  if (!chip) {
    chip = document.createElement('button');
    chip.id = 'chip';
    chip.className = 'chip';
    chip.textContent = 'Stop';
    chip.addEventListener('click', () => {
      if (scanning) stopScanner(); else startScanner();
    });
    document.body.appendChild(chip);
  }
  chip.style.display = 'block';
  chip.textContent = on ? 'Stop' : 'Start';
  if (!on) setTimeout(()=>{ chip.style.display = 'none'; }, 300);
}

// Optional button in HTML with id="scanBtn"
btn?.addEventListener('click', () => scanning ? stopScanner() : startScanner());

// Auto-start on user tap anywhere (mobile-friendly)
document.addEventListener('click', function auto() {
  document.removeEventListener('click', auto);
  startScanner().catch(()=>{ /* handled in HUD */ });
}, { once: true, passive: true });

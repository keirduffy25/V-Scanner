/* ============================================================
   CYBERPUNK V‑SCANNER v1 — "NEON BASE" (GitHub Pages friendly)
   - Camera feed
   - YOLOv8n object detection (web-safe loader with fallbacks)
   - Neon HUD with category colours
   ============================================================ */

const video  = document.getElementById('cam');
const canvas = document.getElementById('hud');
const ctx    = canvas.getContext('2d', { alpha: true });

let model = null;

/* -------- Helpers -------- */
function fitCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
}

function hudColorFor(label = '') {
  const l = label.toLowerCase();
  if (['person','cat','dog','bird'].includes(l)) return '#00b7ff';           // living
  if (['car','bus','truck','bicycle','motorcycle'].includes(l)) return '#ff00ff'; // vehicles
  if (['pizza','apple','sandwich','banana','orange'].includes(l)) return '#ffbf00'; // food
  if (['chair','sofa','couch','bed','table','tv'].includes(l)) return '#00ff66';   // furniture
  return '#00ffff'; // default cyan
}

/* -------- Camera -------- */
async function startCamera() {
  const constraints = {
    audio:false,
    video:{
      facingMode:{ ideal:'environment' },
      width:{ ideal:1280 },
      height:{ ideal:720 }
    }
  };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.setAttribute('playsinline','true');
    video.muted = true;
    await video.play();
    await new Promise(res => {
      if (video.readyState >= 2) return res();
      video.onloadedmetadata = () => res();
    });
    fitCanvasToVideo();
    addEventListener('resize', fitCanvasToVideo);
    console.log('✅ Camera started');
  } catch (err) {
    console.error('❌ Camera error:', err);
    alert('Could not access camera. Open this site directly (not embedded) and allow camera access.');
  }
}

/* -------- Model loader with multiple fallbacks -------- */
async function loadModel() {
  try {
    if (tf?.setBackend) {
      await tf.setBackend('webgl'); // best for mobile GPUs
      await tf.ready();
    }
  } catch (e) {
    console.warn('TF backend warning:', e);
  }

  // If you host yolov8n.onnx in your repo root (recommended for reliability),
  // set LOCAL_MODEL to PUBLIC_BASE + 'yolov8n.onnx' and it will be tried first.
  const LOCAL_MODEL = (window.PUBLIC_BASE || '') + 'yolov8n.onnx';

  const urls = [
    // Local (works if you upload yolov8n.onnx beside index.html)
    LOCAL_MODEL,
    // CDN mirrors (fast to try; sometimes blocked on strict networks)
    'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
    'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx'
  ];

  let lastErr = null;
  for (const url of urls) {
    try {
      console.log('Loading YOLOv8n from:', url);
      const m = await YOLO.load(url); // provided by yolo.min.js
      console.log('✅ Model loaded');
      return m;
    } catch (e) {
      lastErr = e;
      console.warn('Model load failed from', url, e?.message || e);
    }
  }
  console.error('❌ All model sources failed', lastErr);
  alert('Failed to load YOLO model. If using GitHub Pages, consider uploading yolov8n.onnx next to index.html.');
  return null;
}

/* -------- Main loop -------- */
async function loop() {
  if (!model || video.readyState < 2) {
    requestAnimationFrame(loop);
    return;
  }

  let preds = [];
  try {
    preds = await model.detect(video);
  } catch (e) {
    console.error('Detect error:', e);
  }

  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (!preds || preds.length === 0) {
    ctx.save();
    ctx.fillStyle = '#00ffff';
    ctx.textAlign = 'center';
    ctx.font = '18px monospace';
    ctx.fillText('SCANNING…', canvas.width/2, canvas.height/2);
    ctx.restore();
    return requestAnimationFrame(loop);
  }

  const best = preds.reduce((a,b)=> (a.confidence > b.confidence ? a : b));
  const [x,y,w,h] = best.box;
  const label = best.class || 'object';
  const conf  = Math.round((best.confidence||0)*100);
  const color = hudColorFor(label);

  // neon rectangle
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.strokeRect(x,y,w,h);
  ctx.restore();

  // label
  ctx.save();
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.font = '16px monospace';
  ctx.fillText(`${label.toUpperCase()} — ${conf}%`, x + w/2, Math.max(16, y - 8));
  ctx.restore();

  requestAnimationFrame(loop);
}

/* -------- Boot -------- */
(async function init(){
  try {
    await startCamera();
    model = await loadModel();
    if (model) loop();
  } catch (e) {
    console.error('Init error:', e);
    alert('Could not start camera or model.');
  }
})();

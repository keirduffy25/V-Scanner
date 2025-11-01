/* Cyberpunk V-Scanner — MediaPipe Tasks Object Detector (EfficientDet-Lite0)
   - Super-stable in Safari/Chrome on iPad/iPhone
   - No ONNX, no yolo.min.js, no build step
   - Single-target focus (highest score), clean HUD, aligned overlay
*/

const video   = document.getElementById('cam');
const canvas  = document.getElementById('overlay');
const ctx     = canvas.getContext('2d');
const hudCam  = document.getElementById('hud-cam');
const hudModel= document.getElementById('hud-model');
const hudCnt  = document.getElementById('hud-count');
const btn     = document.getElementById('toggleBtn');

let detector = null;
let running  = false;
let rafId    = null;

/* ---------- Canvas sizing (DPR + rotation aware) ---------- */
function fitCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  const dpr  = window.devicePixelRatio || 1;

  // match CSS size
  canvas.style.width  = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  // high-DPI backing store for crisp lines
  canvas.width  = Math.max(1, Math.round(rect.width  * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

  // draw using CSS pixel units
  ctx.setTransform(1,0,0,1,0,0);
  ctx.scale(dpr, dpr);
}

new ResizeObserver(fitCanvasToVideo).observe(video);
window.addEventListener('orientationchange', () => setTimeout(fitCanvasToVideo, 250));
video.addEventListener('loadedmetadata', fitCanvasToVideo);

/* ---------- Camera ---------- */
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  });
  video.srcObject = stream;
  await video.play();
  hudCam.textContent = '✓';
  fitCanvasToVideo();
}

/* ---------- MediaPipe Tasks: ObjectDetector ---------- */
async function loadModel() {
  // Resolve WASM files from CDN
  const vision = await window.FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.6/wasm'
  );

  detector = await window.ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      // Google-hosted model (fast and small)
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/object_detector/' +
        'efficientdet_lite0/float16/1/efficientdet_lite0.tflite'
    },
    runningMode: 'VIDEO',
    scoreThreshold: 0.35, // tweak: lower = more boxes, higher = fewer
    maxResults: 8
  });

  hudModel.textContent = '✓';
}

/* ---------- Drawing ---------- */
function drawDetections(dets) {
  fitCanvasToVideo();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!dets?.length) {
    hudCnt.textContent = '0';
    return;
  }

  // Focus on highest-confidence detection
  const best = dets.slice().sort((a,b)=>(b.categories?.[0]?.score||0) - (a.categories?.[0]?.score||0))[0];
  const cls  = best.categories?.[0]?.categoryName ?? 'object';
  const sc   = best.categories?.[0]?.score ?? 0;

  // MediaPipe returns pixel coords relative to the input frame
  // Map to CSS pixels of the <video> element
  const r    = video.getBoundingClientRect();
  const sx   = r.width  / (video.videoWidth  || r.width);
  const sy   = r.height / (video.videoHeight || r.height);

  const bb   = best.boundingBox;
  const x    = bb.originX * sx;
  const y    = bb.originY * sy;
  const w    = bb.width   * sx;
  const h    = bb.height  * sy;

  // Draw neon box
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(80,255,200,0.95)';
  ctx.shadowColor = 'rgba(80,255,200,0.6)';
  ctx.shadowBlur  = 8;
  ctx.strokeRect(x, y, w, h);

  // Centered label (keeps it from clipping top-left)
  const label = `${cls} ${Math.round(sc*100)}%`;
  ctx.font = '16px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  ctx.textBaseline = 'middle';
  const tw = ctx.measureText(label).width;
  const pad = 6, th = 22;
  const lx = x + w/2 - (tw + pad*2)/2;
  const ly = (y - th - 10 < 8) ? (y + h + 12) : (y - th - 10);

  ctx.fillStyle = 'rgba(0,0,0,0.75)';
  ctx.fillRect(lx, ly, tw + pad*2, th);
  ctx.fillStyle = 'rgba(170,255,230,0.95)';
  ctx.fillText(label, lx + pad, ly + th/2);
  ctx.restore();

  hudCnt.textContent = '1';
}

/* ---------- Main loop ---------- */
function tick() {
  if (!running) return;
  const stamp = performance.now(); // required by detectForVideo
  const result = detector.detectForVideo(video, stamp);
  const dets = result?.detections ?? [];
  drawDetections(dets);
  rafId = requestAnimationFrame(tick);
}

/* ---------- UI ---------- */
btn.addEventListener('click', () => {
  if (!running) {
    running = true;
    btn.textContent = 'Stop';
    rafId && cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  } else {
    running = false;
    btn.textContent = 'Start';
    rafId && cancelAnimationFrame(rafId);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    hudCnt.textContent = '0';
  }
});

/* ---------- Boot ---------- */
(async () => {
  try {
    await startCamera();
    await loadModel();
  } catch (e) {
    console.error(e);
    alert('Init error: ' + (e?.message || e));
  }
})();

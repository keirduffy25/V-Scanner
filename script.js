// Cyberpunk V-Scanner – Version 1
// Basic YOLOv8 object detection with fallback model loading and camera setup

// --- CAMERA SETUP ---
async function startCamera() {
  const video = document.createElement('video');
  video.setAttribute('autoplay', '');
  video.setAttribute('playsinline', '');
  video.style.display = 'none';
  document.body.appendChild(video);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    await new Promise((resolve) => (video.onloadedmetadata = resolve));
    console.log('✅ Camera started');
    return video;
  } catch (err) {
    console.error('❌ Could not access camera:', err);
    alert('Camera access denied or unavailable.');
    throw err;
  }
}

// --- MODEL LOADING ---
async function loadModel() {
  console.log('⏳ Loading YOLOv8 model...');

  // Local model first (works if yolov8n.onnx is in same folder as index.html)
  const LOCAL_MODEL = './yolov8n.onnx';

  // Backup URLs (CDN mirrors)
  const urls = [
    LOCAL_MODEL,
    'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
    'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx'
  ];

  let lastErr = null;
  for (const url of urls) {
    try {
      console.log('Attempting to load YOLOv8n from:', url);
      const model = await YOLO.load(url); // Provided by yolo.min.js
      console.log('✅ Model loaded successfully!');
      return model;
    } catch (err) {
      lastErr = err;
      console.warn('⚠️ Model load failed from:', url, err.message || err);
    }
  }

  console.error('❌ All model sources failed.', lastErr);
  alert('Failed to load YOLO model.\nIf using GitHub Pages, ensure yolov8n.onnx is uploaded next to index.html.');
  return null;
}

// --- MAIN LOOP ---
async function main() {
  const video = await startCamera();
  const model = await loadModel();
  if (!model) return;

  const canvas = document.getElementById('hud');
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  console.log('🚀 Scanner running...');
  requestAnimationFrame(async function detectFrame() {
    const results = await model.detect(video);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const r of results) {
      ctx.strokeStyle = '#00FFFF';
      ctx.lineWidth = 2;
      ctx.strokeRect(r.box[0], r.box[1], r.box[2], r.box[3]);
      ctx.fillStyle = '#00FFFF';
      ctx.font = '14px monospace';
      ctx.fillText(`${r.class} (${Math.round(r.score * 100)}%)`, r.box[0] + 5, r.box[1] + 15);
    }

    requestAnimationFrame(detectFrame);
  });
}

// --- START EVERYTHING ---
main().catch((err) => console.error('Fatal error starting app:', err));

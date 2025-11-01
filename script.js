/* ===========================
   Cyberpunk V-Scanner (YOLOv8 ONNX — no yolo.min.js)
   Works with: index.html containing <video id="cam">, <canvas id="overlay">, <button id="btn">
   Requires: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
   Keep yolov8n.onnx in the REPO ROOT (beside index.html)
   =========================== */

(() => {
  // ---------- Config ----------
  const MODEL_SOURCES = [
    './yolov8n.onnx', // local beside index.html (GitHub Pages friendly)
    'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx',
    'https://cdn.jsdelivr.net/gh/vladmandic/yolo@main/models/yolov8n.onnx'
  ];
  const INPUT_SIZE = 640;      // YOLOv8 default
  const SCORE_THR  = 0.25;     // confidence threshold
  const IOU_THR    = 0.45;     // NMS threshold
  const INFER_EVERY_N_FRAMES = 1; // 1 = every frame, 2 = half framerate, etc.

  // ---------- State ----------
  let session = null;
  let running = false;
  let frameCount = 0;

  // DOM refs
  const els = {};
  function setupDOM() {
    els.video   = document.getElementById('cam')     || document.querySelector('video') || create('video', { id:'cam', playsInline:true, muted:true, autoplay:true });
    els.canvas  = document.getElementById('overlay') || create('canvas', { id:'overlay' });
    els.ctx     = els.canvas.getContext('2d');
    els.btn     = document.getElementById('btn')     || create('button', { id:'btn' }, 'Start');
    els.hud     = document.getElementById('hud')     || create('div', { id:'hud', className:'hud' });

    if (!document.body.contains(els.video))  document.body.appendChild(els.video);
    if (!document.body.contains(els.canvas)) document.body.appendChild(els.canvas);
    if (!document.body.contains(els.btn))    document.body.appendChild(els.btn);
    if (!document.body.contains(els.hud))    document.body.appendChild(els.hud);

    // button wiring
    els.btn.addEventListener('click', () => running ? stop() : start());
  }
  function create(tag, props={}, text) {
    const el = document.createElement(tag);
    Object.assign(el, props);
    if (text) el.textContent = text;
    return el;
  }

  // ---------- HUD ----------
  function resetHUD() {
    els.hud.innerHTML = '';
    addHudLine('Scanner running', true);
  }
  function addHudLine(label, ok) {
    const line = document.createElement('div');
    line.className = 'hud-line';
    line.textContent = `${label} ${ok === undefined ? '' : ok ? '✓' : '✗'}`;
    els.hud.appendChild(line);
  }
  function logLine(txt, ok) { addHudLine(txt, ok); }

  // ---------- Camera ----------
  async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'environment',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    els.video.srcObject = stream;

    await els.video.play().catch(() => {}); // iOS sometimes needs an extra nudge
    await new Promise(r => els.video.onloadedmetadata ? els.video.onloadedmetadata = r : setTimeout(r, 200));

    // Size overlay to video
    sizeOverlay();
    window.addEventListener('resize', sizeOverlay);
    addHudLine('Camera ready', true);
  }
  function sizeOverlay() {
    const w = els.video.videoWidth  || 1280;
    const h = els.video.videoHeight || 720;

    // Match the visible element size
    const rect = els.video.getBoundingClientRect();
    els.canvas.width  = rect.width;
    els.canvas.height = rect.height;

    // Store source video size for mapping later
    els._srcW = w;
    els._srcH = h;
  }

  // ---------- Letterbox helpers (keep boxes aligned) ----------
  function computeLetterbox(srcW, srcH, dst) {
    const scale = Math.min(dst / srcW, dst / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padW = (dst - newW) / 2;
    const padH = (dst - newH) / 2;
    return { scale, newW, newH, padW, padH };
  }

  // ---------- Preprocess (to NCHW float32 [1,3,640,640]) ----------
  const off = document.createElement('canvas');
  const offCtx = off.getContext('2d');
  function preprocess() {
    const vw = els.video.videoWidth;
    const vh = els.video.videoHeight;
    const { newW, newH, padW, padH } = computeLetterbox(vw, vh, INPUT_SIZE);

    off.width = INPUT_SIZE;
    off.height = INPUT_SIZE;

    // clear & draw letterboxed frame
    offCtx.clearRect(0,0,INPUT_SIZE,INPUT_SIZE);
    offCtx.fillStyle = '#000';
    offCtx.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
    offCtx.drawImage(els.video, 0, 0, vw, vh, padW, padH, newW, newH);

    // get pixels
    const img = offCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    const s = 1/255;
    // Convert to CHW
    let p = 0, rOfs = 0, gOfs = INPUT_SIZE*INPUT_SIZE, bOfs = 2*INPUT_SIZE*INPUT_SIZE;
    for (let i=0; i<img.length; i+=4) {
      chw[rOfs + p] = img[i  ] * s;
      chw[gOfs + p] = img[i+1] * s;
      chw[bOfs + p] = img[i+2] * s;
      p++;
    }
    return { tensor: new ort.Tensor('float32', chw, [1,3,INPUT_SIZE,INPUT_SIZE]), padW, padH };
  }

  // ---------- Postprocess (decode + NMS) ----------
  // Works for outputs shaped [1,84,8400] or [1,8400,84]
  function sigmoid(x){ return 1/(1+Math.exp(-x)); }
  function iou(a,b){
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x+a.w, b.x+b.w);
    const y2 = Math.min(a.y+a.h, b.y+b.h);
    const inter = Math.max(0, x2-x1)*Math.max(0, y2-y1);
    return inter / (a.w*a.h + b.w*b.h - inter);
  }
  function nms(boxes, thr) {
    boxes.sort((a,b)=>b.conf-a.conf);
    const keep=[];
    for (const b of boxes) {
      if (keep.every(k=>iou(k,b)<=thr)) keep.push(b);
    }
    return keep;
  }

  function postprocess(output, padW, padH) {
    // Determine layout
    let data, num, attrs;
    const outName = session.outputNames[0];
    const out = output[outName];
    if (!out) return [];
    const shape = out.dims;
    data = out.data;
    if (shape[1] === 84) {        // [1,84,8400]
      attrs = 84;
      num = shape[2];
    } else if (shape[2] === 84) { // [1,8400,84]
      attrs = 84;
      num = shape[1];
    } else {
      console.warn('Unexpected YOLO output shape:', shape);
      return [];
    }

    // Decode boxes
    const boxes=[];
    for (let i=0;i<num;i++){
      const base = (shape[1]===84) ? i : i*attrs;
      const cx = data[base+0], cy = data[base+1], w = data[base+2], h = data[base+3];

      // class scores start at index 4
      let bestC = 0, bestS = 0;
      for (let c=4;c<attrs;c++){
        const score = data[base+c];
        if (score>bestS) { bestS = score; bestC = c-4; }
      }
      const conf = sigmoid(bestS);
      if (conf < SCORE_THR) continue;

      // map back to INPUT_SIZE (remove padding) then to canvas size
      let x = cx - w/2;
      let y = cy - h/2;

      // undo letterbox scale
      x = (x - padW);
      y = (y - padH);

      const { newW, newH } = computeLetterbox(els._srcW, els._srcH, INPUT_SIZE);

      const scaleX = (els.canvas.width  / newW);
      const scaleY = (els.canvas.height / newH);

      x *= scaleX;
      y *= scaleY;
      const ww = w * scaleX;
      const hh = h * scaleY;

      // Also shift because we letterboxed within INPUT_SIZE
      const shiftX = (els.canvas.width  - newW * scaleX) / 2;
      const shiftY = (els.canvas.height - newH * scaleY) / 2;

      boxes.push({ x: x + shiftX, y: y + shiftY, w: ww, h: hh, conf, cls: bestC });
    }
    return nms(boxes, IOU_THR);
  }

  // ---------- Inference loop ----------
  async function tick() {
    if (!running) return;

    frameCount++;
    const doInfer = (frameCount % INFER_EVERY_N_FRAMES) === 0;

    if (doInfer && session) {
      const { tensor, padW, padH } = preprocess();
      const feeds = {};
      feeds[session.inputNames[0]] = tensor;

      const output = await session.run(feeds);
      const dets = postprocess(output, padW, padH);

      draw(dets);
      updateSmallHud(dets.length);
    }

    requestAnimationFrame(tick);
  }

  function updateSmallHud(n) {
    // Keep first 3 lines concise
    const lines = els.hud.querySelectorAll('.hud-line');
    if (lines.length > 0) lines[0].textContent = `Scanner running ✓`;
    if (lines.length > 1) lines[1].textContent = `Camera ready ✓`;
    if (lines.length > 2) lines[2].textContent = `Model loaded ✓`;
    // Ensure a 4th status about detections
    let dLine = els.hud.querySelector('.hud-dets');
    if (!dLine) {
      dLine = document.createElement('div');
      dLine.className = 'hud-line hud-dets';
      els.hud.appendChild(dLine);
    }
    dLine.textContent = `Detections: ${n}`;
  }

  // ---------- Drawing ----------
  function draw(dets) {
    const ctx = els.ctx;
    ctx.clearRect(0,0,els.canvas.width, els.canvas.height);

    // Draw all
    ctx.lineWidth = 2;
    for (const d of dets) {
      ctx.strokeStyle = 'rgba(0,255,170,0.9)';
      ctx.fillStyle   = 'rgba(0,255,170,0.15)';
      roundRect(ctx, d.x, d.y, d.w, d.h, 8);
      ctx.stroke();
      ctx.fill();

      const label = `${(d.conf*100|0)}%`;
      const tw = ctx.measureText ? ctx.measureText(label).width + 10 : 60;
      const th = 18;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(d.x, Math.max(0,d.y-th), tw, th);
      ctx.fillStyle = '#0ff';
      ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
      ctx.fillText(label, d.x+5, Math.max(12, d.y-4));
    }
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x+r, y);
    ctx.arcTo(x+w, y, x+w, y+h, r);
    ctx.arcTo(x+w, y+h, x, y+h, r);
    ctx.arcTo(x, y+h, x, y, r);
    ctx.arcTo(x, y, x+w, y, r);
    ctx.closePath();
  }

  // ---------- Model ----------
  async function loadModel() {
    const ortLib = window.ort;
    if (!ortLib) throw new Error('ONNX Runtime Web (ort.min.js) not loaded');

    let lastErr = null;
    for (const url of MODEL_SOURCES) {
      try {
        logLine(`Attempting YOLOv8 model: ${url}`);
        session = await ortLib.InferenceSession.create(url, {
          executionProviders: ['wasm'],
        });
        logLine('Model loaded', true);
        return;
      } catch (e) {
        lastErr = e;
        console.warn(`Model load failed from ${url}:`, e);
        logLine(`Failed to load from: ${url}`, false);
      }
    }
    throw new Error('❌ All YOLOv8 sources failed. Ensure yolov8n.onnx is beside index.html.');
  }

  // ---------- Start/Stop ----------
  async function start() {
    try {
      setupDOM();
      resetHUD();

      if (!session) {
        logLine('Loading model ⏳');
        await loadModel();
      }
      await initCamera();

      running = true;
      els.btn.textContent = 'Stop';
      tick();
    } catch (e) {
      console.error(e);
      alert(e.message || String(e));
      stop();
    }
  }

  function stop() {
    running = false;
    els.btn.textContent = 'Start';
    els.ctx.clearRect(0,0,els.canvas.width, els.canvas.height);
    // keep camera stream to allow instant restart
  }

  // ---------- Auto-init small HUD + keep canvas over video ----------
  document.addEventListener('DOMContentLoaded', () => {
    setupDOM();
    // ensure overlay sits exactly on top of the video element
    const style = (el, s) => Object.assign(el.style, s);
    style(els.video,  { width:'100vw', height:'100vh', objectFit:'cover', position:'fixed', left:0, top:0 });
    style(els.canvas, { width:'100vw', height:'100vh', position:'fixed', left:0, top:0, pointerEvents:'none' });
    style(els.hud,    { position:'fixed', left:'12px', top:'12px', padding:'8px 10px',
                        background:'rgba(0,0,0,.35)', color:'#bff', font:'12px ui-monospace,monospace',
                        border:'1px solid rgba(0,255,170,.3)', borderRadius:'10px', backdropFilter:'blur(2px)' });
    style(els.btn,    { position:'fixed', left:'50%', bottom:'22px', transform:'translateX(-50%)',
                        padding:'10px 18px', borderRadius:'12px', border:'1px solid #0aa', background:'rgba(0,0,0,.45)',
                        color:'#bff', font:'14px ui-monospace,monospace' });

    addHudLine('Camera ready', false); // will become ✓ after start()

    // Optional: auto-start if camera is already granted
    // start();
  });
})();

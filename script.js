/* =========================================================================
   Cyberpunk V-Scanner (Version 1)
   - YOLOv8n (ONNX) + onnxruntime-web
   - iOS/iPadOS friendly camera constraints
   - Robust model URL resolution (local first, then backups)
   - HUD status panel
   - EXTRA: Output shape logging + safe fallback box if decode fails/empty
   ========================================================================= */

(() => {
  const W = 640, H = 640;               // model input size (square)
  const SCORE_THR = 0.25;               // confidence threshold
  const IOU_THR = 0.45;                 // NMS threshold

  // Elements
  const video  = document.getElementById('cam');
  const canvas = document.getElementById('overlay');
  const ctx    = canvas.getContext('2d');

  // HUD (status text panel, drawn top-left)
  const hud = [];
  function hudPush(line) {
    hud.unshift(line);
    if (hud.length > 10) hud.pop();
  }
  function drawHUD() {
    const pad = 8;
    const lineH = 16;
    const boxW = 250, boxH = pad * 2 + lineH * hud.length;

    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = 'rgba(0,20,25,0.6)';
    ctx.strokeStyle = '#00e6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(10, 10, boxW, boxH, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#9ff';
    let y = 10 + pad + 12;
    for (const line of [...hud].reverse()) {
      ctx.fillText(line, 18, y);
      y += lineH;
    }
    ctx.restore();
  }

  // --- Utilities -----------------------------------------------------------
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // Simple NMS for [x1,y1,x2,y2,score,cls]
  function nms(boxes, iouThr) {
    boxes.sort((a,b) => b[4] - a[4]);
    const keep = [];
    const iou = (a, b) => {
      const xa = Math.max(a[0], b[0]);
      const ya = Math.max(a[1], b[1]);
      const xb = Math.min(a[2], b[2]);
      const yb = Math.min(a[3], b[3]);
      const w = Math.max(0, xb - xa);
      const h = Math.max(0, yb - ya);
      const inter = w * h;
      const ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
      return ua > 0 ? inter / ua : 0;
    };
    for (const b of boxes) {
      let ok = true;
      for (const k of keep) { if (iou(b,k) > iouThr) { ok = false; break; } }
      if (ok) keep.push(b);
    }
    return keep;
  }

  // Draw detections
  function drawDetections(dets) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.lineWidth = 2;
    ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (const d of dets) {
      const [x1,y1,x2,y2,score,clsName] = d;
      ctx.strokeStyle = '#00fff2';
      ctx.shadowColor = '#00fff2';
      ctx.shadowBlur = 8;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${clsName ?? 'obj'} ${(score*100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = 'rgba(0,25,30,0.85)';
      ctx.fillRect(x1, y1 - 18, tw, 18);
      ctx.fillStyle = '#9ff';
      ctx.fillText(label, x1 + 4, y1 - 4);
    }
    ctx.restore();
  }

  // --- Model URL resolution ------------------------------------------------
  // 1) local yolov8n.onnx (same folder as index.html)
  // 2) your repo raw as backup (edit if you change repo name)
  const LOCAL = 'yolov8n.onnx';
  const BACKUP = 'https://raw.githubusercontent.com/keirduffy25/V-Scanner/main/yolov8n.onnx';
  const urls = [LOCAL, BACKUP];

  // --- Camera --------------------------------------------------------------
  async function setupCamera() {
    // iOS/iPadOS friendly constraints
    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width:  { ideal: 1280 },
        height: { ideal: 720 },
      }
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    // Fit canvas to video display size
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    canvas.width = vw;
    canvas.height = vh;
    return { vw, vh };
  }

  // Preprocess frame into 1x3x640x640 float tensor
  const work = document.createElement('canvas');
  work.width = W; work.height = H;
  const wctx = work.getContext('2d', { willReadFrequently: true });

  function preprocess(scaleInfo) {
    // letterbox to 640x640
    const { vw, vh } = scaleInfo;
    const r = Math.min(W / vw, H / vh);
    const nw = Math.round(vw * r);
    const nh = Math.round(vh * r);
    const dx = Math.floor((W - nw) / 2);
    const dy = Math.floor((H - nh) / 2);

    wctx.clearRect(0,0,W,H);
    wctx.drawImage(video, 0,0,vw,vh, dx,dy,nw,nh);

    const img = wctx.getImageData(0,0,W,H).data;
    // convert to float32 CHW normalized 0..1
    const out = new Float32Array(1 * 3 * H * W);
    let p = 0, rIdx = 0, gIdx = H*W, bIdx = 2*H*W;
    for (let i = 0; i < img.length; i += 4) {
      const r8 = img[i] / 255;
      const g8 = img[i+1] / 255;
      const b8 = img[i+2] / 255;
      out[rIdx++] = r8;
      out[gIdx++] = g8;
      out[bIdx++] = b8;
    }
    // store for reverse scaling when drawing
    scaleInfo.r = r; scaleInfo.dx = dx; scaleInfo.dy = dy;
    return out;
  }

  // --- YOLOv8 decode (handles common shapes) ------------------------------
  // Accepts ONNX output tensor (ort.Tensor) and scaleInfo {r,dx,dy,vw,vh}
  // Returns array of [x1,y1,x2,y2,score,clsName]
  function decode(outputs, scaleInfo) {
    const outName = session.outputNames[0];
    const out = outputs[outName]; // ort.Tensor
    const d = out.dims;           // e.g., [1,84,8400] or [1,8400,84] or [1,25200,85]
    const data = out.data;        // Float32Array

    // Map COCO80 ids to short labels (minimal set to prove working)
    const names = [
      'person','bicycle','car','motorbike','aeroplane','bus','train','truck',
      'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
      'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
      'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
      'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
      'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
      'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
      'donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet',
      'tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave',
      'oven','toaster','sink','refrigerator','book','clock','vase','scissors',
      'teddy bear','hair drier','toothbrush'
    ];

    let boxes = [];

    // Layout A: [1, 84, 8400]  -> (x,y,w,h, obj, 80 classes) across 8400 anchors
    if (d.length === 3 && d[1] >= 84) {
      const C = d[1];            // 84 or 85
      const N = d[2];
      for (let i = 0; i < N; i++) {
        const off = i;                       // column index
        const x = data[0*N + off];
        const y = data[1*N + off];
        const w = data[2*N + off];
        const h = data[3*N + off];
        const obj = (C > 84) ? data[4*N + off] : 1.0; // some exports include obj
        let best = 0, bi = -1;
        const clsStart = (C > 84) ? 5 : 4;
        for (let c = clsStart; c < C; c++) {
          const v = data[c*N + off];
          if (v > best) { best = v; bi = c - clsStart; }
        }
        const score = best * obj;
        if (score >= SCORE_THR && bi >= 0) {
          // convert xywh->xyxy then unletterbox
          const cx = x, cy = y, ww = w, hh = h;
          let x1 = cx - ww/2, y1 = cy - hh/2, x2 = cx + ww/2, y2 = cy + hh/2;
          // coords are already in input scale (0..640)
          // unletterbox back to video space
          x1 = (x1 - scaleInfo.dx) / scaleInfo.r;
          y1 = (y1 - scaleInfo.dy) / scaleInfo.r;
          x2 = (x2 - scaleInfo.dx) / scaleInfo.r;
          y2 = (y2 - scaleInfo.dy) / scaleInfo.r;

          x1 = clamp(x1, 0, canvas.width); y1 = clamp(y1, 0, canvas.height);
          x2 = clamp(x2, 0, canvas.width); y2 = clamp(y2, 0, canvas.height);

          boxes.push([x1,y1,x2,y2,score,names[bi] || `c${bi}`]);
        }
      }
    }
    // Layout B: [1, 8400, 84/85] or [1, 25200, 85]
    else if (d.length === 3 && (d[2] >= 84 || d[2] === 85)) {
      const N = d[1];
      const C = d[2];
      for (let i = 0; i < N; i++) {
        const base = i*C;
        const cx = data[base + 0];
        const cy = data[base + 1];
        const ww = data[base + 2];
        const hh = data[base + 3];
        const obj = (C > 84) ? data[base + 4] : 1.0;
        let best = 0, bi = -1;
        const clsStart = (C > 84) ? 5 : 4;
        for (let c = clsStart; c < C; c++) {
          const v = data[base + c];
          if (v > best) { best = v; bi = c - clsStart; }
        }
        const score = best * obj;
        if (score >= SCORE_THR && bi >= 0) {
          let x1 = cx - ww/2, y1 = cy - hh/2, x2 = cx + ww/2, y2 = cy + hh/2;
          x1 = (x1 - scaleInfo.dx) / scaleInfo.r;
          y1 = (y1 - scaleInfo.dy) / scaleInfo.r;
          x2 = (x2 - scaleInfo.dx) / scaleInfo.r;
          y2 = (y2 - scaleInfo.dy) / scaleInfo.r;

          x1 = clamp(x1, 0, canvas.width); y1 = clamp(y1, 0, canvas.height);
          x2 = clamp(x2, 0, canvas.width); y2 = clamp(y2, 0, canvas.height);

          boxes.push([x1,y1,x2,y2,score,names[bi] || `c${bi}`]);
        }
      }
    } else {
      console.warn('Unknown output shape', d);
    }

    boxes = nms(boxes, IOU_THR);
    return boxes;
  }

  // --- Main ---------------------------------------------------------------
  let session, inputName;
  let scaleInfo = { vw: 1280, vh: 720, r: 1, dx: 0, dy: 0 };
  let running = false;

  async function loadModel() {
    hudPush('Loading YOLOv8 (.onnx)…');
    // Try local first, then backups
    let lastErr = null;
    for (const url of urls) {
      try {
        console.log('Loading YOLOv8 from:', url);
        const m = await YOLO.load(url); // provided by yolo.min.js
        console.log('✅ Model loaded via yolo.min.js wrapper');
        return m; // wrapper returns an object w/ .session, .inputName or similar
      } catch (e) {
        lastErr = e;
        console.warn('Model load failed from', url, e?.message || e);
      }
    }
    throw lastErr ?? new Error('All model URLs failed');
  }

  // If you’re not using the wrapper’s helpers, fall back to ort directly
  async function loadModelORT(url) {
    const ort = window.ort;
    session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    inputName = session.inputNames[0];
    return { session, inputName };
  }

  async function start() {
    hudPush('V-Scanner initialising…');

    // Camera
    try {
      const { vw, vh } = await setupCamera();
      scaleInfo.vw = vw; scaleInfo.vh = vh;
      hudPush('Camera ready ✓');
    } catch (e) {
      hudPush('⚠ Camera not available');
      throw e;
    }

    // Model
    try {
      // Prefer direct ORT session to avoid wrapper variability
      // First try local then backup:
      let lastErr = null, ok = false;
      for (const url of urls) {
        try {
          console.log('ORT create from:', url);
          const ort = window.ort;
          session = await ort.InferenceSession.create(url, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
          });
          inputName = session.inputNames[0];
          ok = true;
          break;
        } catch (e) { lastErr = e; }
      }
      if (!ok) throw lastErr;
      hudPush('Model loaded ✓');
    } catch (e) {
      console.error(e);
      alert('Failed to load YOLO model. If using GitHub Pages, ensure yolov8n.onnx sits next to index.html.');
      return;
    }

    // ORT ready
    hudPush('ORT ready ✓');

    running = true;
    requestAnimationFrame(loop);
  }

  async function loop() {
    if (!running) return;

    // HUD header
    ctx.clearRect(0,0,canvas.width, canvas.height);
    hudPush('Scanner running ✓');

    // Preprocess
    const inputData = preprocess(scaleInfo);

    // Create tensor: [1,3,640,640]
    const tensor = new ort.Tensor('float32', inputData, [1,3,H,W]);

    // Inference
    let outputs;
    try {
      outputs = await session.run({ [inputName]: tensor });
    } catch (e) {
      console.error('Inference failed:', e);
      hudPush('Inference error');
      drawHUD();
      return requestAnimationFrame(loop);
    }

    // --- EXTRA DEBUG: log output structure
    const outName = session.outputNames[0];
    const out = outputs[outName];
    console.log('Model outputs:', outputs);
    console.log('Output tensor dims:', out?.dims);
    if (out?.data) {
      console.log('Output tensor sample:', out.data.slice(0, 20));
    }

    // Decode + draw (with safe fallback)
    let dets = [];
    try {
      dets = decode(outputs, scaleInfo) || [];
    } catch (err) {
      console.warn('Decode failed:', err);
      hudPush('Decode failed, showing test box');
    }

    console.log('Detections:', dets.length);
    hudPush(`Detections: ${dets.length}`);

    if (dets.length === 0) {
      // Fallback sanity box so we know drawing works
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 5;
      ctx.strokeRect(50, 50, 200, 200);
      ctx.font = '20px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = 'red';
      ctx.fillText('No detections', 60, 40);
    } else {
      drawDetections(dets);
    }

    drawHUD();
    requestAnimationFrame(loop);
  }

  // Kick off once libraries have loaded
  window.addEventListener('load', () => {
    // yolo.min.js must be loaded before this file (index.html handles that),
    // but we primarily use onnxruntime-web directly here.
    if (!window.ort) {
      alert('onnxruntime-web not loaded.');
      return;
    }
    // iOS autoplay policies
    video.setAttribute('playsinline', '');
    start().catch(err => {
      console.error(err);
      hudPush('Init error');
      drawHUD();
    });
  });
})();

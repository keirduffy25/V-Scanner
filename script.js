/* =========================================================================
   Cyberpunk V-Scanner (Version 1) — Click-to-start
   ========================================================================= */

(() => {
  const W = 640, H = 640;
  const SCORE_THR = 0.25;
  const IOU_THR = 0.45;

  // Elements
  const video   = document.getElementById('cam');
  const canvas  = document.getElementById('overlay');
  const ctx     = canvas.getContext('2d');
  const btnGo   = document.getElementById('activate');
  const btnStop = document.getElementById('stop');

  // HUD
  const hud = [];
  function hudPush(line) {
    hud.unshift(line);
    if (hud.length > 10) hud.pop();
  }
  function drawHUD() {
    const pad = 8, lineH = 16, boxW = 260, boxH = pad*2 + lineH*hud.length;
    ctx.save();
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = 'rgba(0,20,25,0.6)';
    ctx.strokeStyle = '#00e6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect?.(10,10,boxW,boxH,8);
    if (!ctx.roundRect) { ctx.rect(10,10,boxW,boxH); }
    ctx.fill(); ctx.stroke();
    ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = '#9ff';
    let y = 10 + pad + 12;
    for (const line of [...hud].reverse()) { ctx.fillText(line, 18, y); y += lineH; }
    ctx.restore();
  }

  // Utils
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function nms(boxes, iouThr) {
    boxes.sort((a,b) => b[4]-a[4]);
    const keep=[]; const iou=(a,b)=>{
      const xa=Math.max(a[0],b[0]), ya=Math.max(a[1],b[1]);
      const xb=Math.min(a[2],b[2]), yb=Math.min(a[3],b[3]);
      const w=Math.max(0,xb-xa), h=Math.max(0,yb-ya);
      const inter=w*h, ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;
      return ua>0?inter/ua:0;
    };
    for (const b of boxes) {
      let ok=true; for (const k of keep) { if (iou(b,k)>iouThr) { ok=false; break; } }
      if (ok) keep.push(b);
    }
    return keep;
  }
  function drawDetections(dets) {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.save(); ctx.lineWidth=2; ctx.font='14px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (const d of dets) {
      const [x1,y1,x2,y2,score,clsName] = d;
      ctx.strokeStyle='#00fff2'; ctx.shadowColor='#00fff2'; ctx.shadowBlur=8;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const label=`${clsName??'obj'} ${(score*100).toFixed(0)}%`;
      const tw=ctx.measureText(label).width+8;
      ctx.fillStyle='rgba(0,25,30,0.85)'; ctx.fillRect(x1,y1-18,tw,18);
      ctx.fillStyle='#9ff'; ctx.fillText(label,x1+4,y1-4);
    }
    ctx.restore();
  }

  // Model URLs (local first, then backup raw)
  const urls = [
    'yolov8n.onnx',
    'https://raw.githubusercontent.com/keirduffy25/V-Scanner/main/yolov8n.onnx'
  ];

  // Camera
  async function setupCamera() {
    const constraints = {
      audio: false,
      video: { facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720} }
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    canvas.width = vw; canvas.height = vh;
    return { vw, vh };
  }

  // Preprocess
  const work = document.createElement('canvas'); work.width=W; work.height=H;
  const wctx = work.getContext('2d', { willReadFrequently:true });
  function preprocess(scaleInfo) {
    const { vw, vh } = scaleInfo;
    const r = Math.min(W/vw, H/vh), nw=Math.round(vw*r), nh=Math.round(vh*r);
    const dx=Math.floor((W-nw)/2), dy=Math.floor((H-nh)/2);
    wctx.clearRect(0,0,W,H);
    wctx.drawImage(video,0,0,vw,vh,dx,dy,nw,nh);
    const img = wctx.getImageData(0,0,W,H).data;
    const out = new Float32Array(1*3*H*W);
    let rI=0,gI=H*W,bI=2*H*W;
    for (let i=0;i<img.length;i+=4) {
      out[rI++] = img[i]   /255;
      out[gI++] = img[i+1] /255;
      out[bI++] = img[i+2] /255;
    }
    scaleInfo.r=r; scaleInfo.dx=dx; scaleInfo.dy=dy;
    return out;
  }

  // Decode
  const names = [ 'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie',
    'suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
    'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut',
    'cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
  ];
  function decode(outputs, scaleInfo, session, canvas) {
    const outName = session.outputNames[0];
    const out = outputs[outName];
    const d = out.dims;
    const data = out.data;
    let boxes=[];
    // [1,84,8400]
    if (d.length===3 && d[1] >= 84) {
      const C=d[1], N=d[2];
      const clsStart=(C>84)?5:4, objHas=(C>84);
      for (let i=0;i<N;i++){
        const x=data[0*N+i], y=data[1*N+i], w=data[2*N+i], h=data[3*N+i];
        const obj=objHas?data[4*N+i]:1;
        let best=0, bi=-1;
        for (let c=clsStart;c<C;c++){ const v=data[c*N+i]; if (v>best){best=v;bi=c-clsStart;} }
        const score=best*obj; if (score<SCORE_THR || bi<0) continue;
        let x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+h/2;
        x1=(x1-scaleInfo.dx)/scaleInfo.r; y1=(y1-scaleInfo.dy)/scaleInfo.r;
        x2=(x2-scaleInfo.dx)/scaleInfo.r; y2=(y2-scaleInfo.dy)/scaleInfo.r;
        x1=clamp(x1,0,canvas.width); y1=clamp(y1,0,canvas.height);
        x2=clamp(x2,0,canvas.width); y2=clamp(y2,0,canvas.height);
        boxes.push([x1,y1,x2,y2,score,names[bi]||`c${bi}`]);
      }
    }
    // [1,8400,84/85] or [1,25200,85]
    else if (d.length===3 && (d[2] >= 84 || d[2] === 85)) {
      const N=d[1], C=d[2]; const clsStart=(C>84)?5:4, objHas=(C>84);
      for (let i=0;i<N;i++){
        const base=i*C;
        const x=data[base+0], y=data[base+1], w=data[base+2], h=data[base+3];
        const obj=objHas?data[base+4]:1;
        let best=0, bi=-1;
        for (let c=clsStart;c<C;c++){ const v=data[base+c]; if (v>best){best=v;bi=c-clsStart;} }
        const score=best*obj; if (score<SCORE_THR || bi<0) continue;
        let x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+h/2;
        x1=(x1-scaleInfo.dx)/scaleInfo.r; y1=(y1-scaleInfo.dy)/scaleInfo.r;
        x2=(x2-scaleInfo.dx)/scaleInfo.r; y2=(y2-scaleInfo.dy)/scaleInfo.r;
        x1=clamp(x1,0,canvas.width); y1=clamp(y1,0,canvas.height);
        x2=clamp(x2,0,canvas.width); y2=clamp(y2,0,canvas.height);
        boxes.push([x1,y1,x2,y2,score,names[bi]||`c${bi}`]);
      }
    } else {
      console.warn('Unknown output shape', d);
    }
    return nms(boxes, IOU_THR);
  }

  // State
  let session, inputName, running=false, rafId=null;
  let scaleInfo={ vw:1280, vh:720, r:1, dx:0, dy:0 };

  async function loadORTSession() {
    let lastErr=null;
    for (const url of urls) {
      try {
        const ort=window.ort;
        session = await ort.InferenceSession.create(url, {
          executionProviders:['wasm'],
          graphOptimizationLevel:'all'
        });
        inputName = session.inputNames[0];
        return;
      } catch(e){ lastErr=e; }
    }
    throw lastErr ?? new Error('All model URLs failed');
  }

  async function start() {
    if (running) return;
    hud.length = 0;
    hudPush('Tap Activate Scanner to begin…');
    drawHUD();

    try {
      const { vw, vh } = await setupCamera();
      scaleInfo.vw=vw; scaleInfo.vh=vh;
      hudPush('Camera ready ✓');
    } catch(e) {
      hudPush('⚠ Camera not available'); drawHUD(); return;
    }

    try {
      hudPush('Loading model…'); drawHUD();
      await loadORTSession();
      hudPush('Model loaded ✓');
      hudPush('ORT ready ✓');
    } catch(e) {
      console.error(e);
      alert('Failed to load YOLO model. Ensure yolov8n.onnx sits next to index.html.');
      return;
    }

    running = true;
    btnGo.style.display='none';
    btnStop.style.display='inline-block';
    rafId = requestAnimationFrame(loop);
  }

  function stop() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    hud.length = 0;
    hudPush('Scanner stopped');
    drawHUD();
    btnStop.style.display='none';
    btnGo.style.display='inline-block';
    // Keep camera stream alive to allow quick restart (iOS is picky).
  }

  async function loop() {
    if (!running) return;

    // HUD banner
    ctx.clearRect(0,0,canvas.width,canvas.height);
    hudPush('Scanner running ✓');

    // Preprocess
    const inputData = preprocess(scaleInfo);
    const tensor = new ort.Tensor('float32', inputData, [1,3,H,W]);

    // Inference
    let outputs;
    try { outputs = await session.run({ [inputName]: tensor }); }
    catch (e) {
      console.error('Inference failed:', e);
      hudPush('Inference error'); drawHUD();
      return rafId = requestAnimationFrame(loop);
    }

    // Decode + Draw with safe fallback
    let dets = [];
    try { dets = decode(outputs, scaleInfo, session, canvas) || []; }
    catch (err) { console.warn('Decode failed:', err); hudPush('Decode failed'); }

    if (dets.length === 0) {
      // Visible fallback to confirm drawing works
      ctx.strokeStyle='red'; ctx.lineWidth=5;
      ctx.strokeRect(50,50,200,200);
      ctx.font='20px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle='red'; ctx.fillText('No detections', 60, 40);
    } else {
      drawDetections(dets);
    }
    hudPush(`Detections: ${dets.length}`);
    drawHUD();

    rafId = requestAnimationFrame(loop);
  }

  // Wire buttons AFTER libraries load (no auto-start)
  window.addEventListener('load', () => {
    if (!window.ort) {
      alert('onnxruntime-web not loaded.');
      return;
    }
    video.setAttribute('playsinline','');
    hudPush('Tap Activate Scanner to begin…');
    drawHUD();
    btnGo.addEventListener('click', start, { passive: true });
    btnStop.addEventListener('click', stop,  { passive: true });
  });
})();

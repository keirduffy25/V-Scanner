// Cyberpunk V-Scanner — Stable V1
// YOLOv8n (ONNX) + iPad camera fixes + absolute model URL + diagnostics + SCANNING overlay.

// ---------- tiny on-screen logger (bottom-left) ----------
const hud = (() => {
  const el = document.createElement('div');
  el.style.cssText = `
    position:fixed;left:8px;bottom:8px;max-width:90vw;z-index:9999;
    background:#000c;color:#0ff;border:1px solid #044;box-shadow:0 0 10px #0ff;
    padding:8px 10px;font:12px ui-monospace,monospace;line-height:1.45;white-space:pre-wrap`;
  el.textContent = 'Starting V-Scanner…';
  addEventListener('DOMContentLoaded', () => document.body.appendChild(el));
  return { set:(t)=>el.textContent=t, add:(t)=>el.textContent+=`\n${t}` };
})();

function inIframe(){ try{return self!==top;}catch(_){return true;} }

// ---------- DOM refs ----------
const canvas = document.getElementById('hud');
const ctx = canvas.getContext('2d', { alpha: true });
const overlay = document.getElementById('scanningOverlay');
const nosupport = document.getElementById('nosupport');

// ---------- absolute model URL + mirrors ----------
const MODEL_URLS = [
  'https://keirduffy25.github.io/V-Scanner/yolov8n.onnx', // your hosted copy
  'https://cdn.jsdelivr.net/gh/vladmandic/yolo/models/yolov8n.onnx',
  'https://raw.githubusercontent.com/vladmandic/yolo/main/models/yolov8n.onnx'
];

// Probe URLs so we can show clear reasons on iPad
async function probe(url){
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-1023' } });
    return { ok: res.ok, status: res.status, url: res.url };
  } catch (e) {
    return { ok:false, status:'network-error', url, err:e.message||String(e) };
  }
}

// Camera
async function startCamera(){
  if(!navigator.mediaDevices?.getUserMedia){
    nosupport.style.display = 'block';
    throw new Error('Camera API not available in this browser.');
  }
  const v = document.createElement('video');
  v.setAttribute('playsinline',''); v.setAttribute('autoplay',''); v.muted = true; v.style.display='none';
  document.body.appendChild(v);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video:{ facingMode:{ ideal:'environment' }, width:{ideal:1280}, height:{ideal:720} },
      audio:false
    });
    v.srcObject = stream;
    await new Promise(r => (v.readyState>=2 ? r() : (v.onloadedmetadata=r)));
    return v;
  } catch(e){
    const hint = 'Camera blocked.\nSafari: AA ▸ Website Settings ▸ Camera ▸ Allow.\nChrome: ⋮ ▸ Site Settings ▸ Camera ▸ Allow.';
    throw new Error(`${e.name||'getUserMedia'}: ${e.message||e}. ${hint}`);
  }
}

// YOLO loader with diagnostics
async function loadModel(){
  if(!window.YOLO?.load) throw new Error('yolo.min.js not loaded before script.js');

  for(const url of MODEL_URLS){
    const p = await probe(url);
    hud.add(`Probe: ${p.url} → ${p.ok ? '200 OK' : p.status}`);
    if(!p.ok) continue;
    try {
      hud.add(`Loading model from: ${p.url}`);
      const m = await YOLO.load(p.url);
      hud.add('✅ YOLO model loaded');
      return m;
    } catch(e){
      hud.add(`❌ YOLO.load() failed: ${e.message||e}`);
    }
  }
  throw new Error('Failed to load YOLO model.\nCheck this opens: https://keirduffy25.github.io/V-Scanner/yolov8n.onnx');
}

// Canvas sizing
function fitCanvasToVideo(c,v){ c.width = v.videoWidth||1280; c.height = v.videoHeight||720; }

// Drawing helpers
function glowRect(ctx,x,y,w,h,color='#0ff'){
  ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=2; ctx.shadowColor=color; ctx.shadowBlur=10;
  ctx.strokeRect(x,y,w,h); ctx.restore();
}
function label(ctx, text, x, y, color='#0ff'){
  ctx.save(); ctx.font='14px ui-monospace,monospace'; ctx.fillStyle=color; ctx.textAlign='center';
  ctx.shadowColor=color; ctx.shadowBlur=6; ctx.fillText(text, x, y); ctx.restore();
}

// ---------- MAIN ----------
(async () => {
  hud.set('V-Scanner initializing…');
  if(inIframe()){
    hud.add('⚠️ Embedded preview detected. Open directly:\nhttps://keirduffy25.github.io/V-Scanner/');
  }

  try {
    overlay.style.display = 'block';         // show “SCANNING…” during startup
    const video = await startCamera();
    hud.add('✅ Camera ready');

    fitCanvasToVideo(canvas, video);
    addEventListener('resize', () => fitCanvasToVideo(canvas, video));

    hud.add('Loading YOLO model…');
    const model = await loadModel();

    hud.add('🚀 Scanner running');
    // render loop
    async function loop(){
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      let preds = [];
      try { preds = await model.detect(video); } catch(_){ /* ignore short hiccups */ }

      if(preds && preds.length){
        overlay.style.display = 'none'; // hide overlay once we have detections
        const best = preds.reduce((a,b)=> (a.score>b.score ? a : b));
        const [x,y,w,h] = best.box;
        glowRect(ctx, x,y,w,h, '#00ffff');
        label(ctx, `${best.class.toUpperCase()} — ${Math.round(best.score*100)}%`,
              x + w/2, Math.max(16, y-8), '#00ffff');
      } else {
        overlay.style.display = 'block'; // no detections yet → show overlay
      }

      requestAnimationFrame(loop);
    }
    loop();
  } catch(err){
    console.error(err);
    hud.add(`❌ Error: ${err.message||err}`);
    nosupport.style.display = 'block';
    alert(err.message || String(err));
  }
})();

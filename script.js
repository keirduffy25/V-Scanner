<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Cyberpunk V-Scanner</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <!-- HUD canvas (script will resize & draw on it) -->
  <canvas id="hud"></canvas>

  <!-- Simple status box so you never see a blank screen -->
  <div id="boot" style="
    position:fixed;left:8px;bottom:8px;z-index:99999;
    background:#001016cc;border:1px solid #00ffff55;box-shadow:0 0 10px #00ffff66;
    color:#0ff;padding:8px 10px;font:12px ui-monospace,monospace;border-radius:6px;
    white-space:pre-line;max-width:68vw;">
    Booting V-Scanner…
  </div>

  <script>
    // -------- mini logger shown on page --------
    const bootBox = document.getElementById('boot');
    const bootLog = (m) => bootBox.textContent += (bootBox.textContent ? '\n' : '') + m;

    // -------- load script helper (ordered, with onload) --------
    function loadScript(src, { async=true, onload=null, onerror=null } = {}) {
      const s = document.createElement('script');
      s.src = src;
      s.async = async;
      s.crossOrigin = 'anonymous';
      if (onload)  s.onload  = onload;
      if (onerror) s.onerror = onerror;
      document.body.appendChild(s);
      return s;
    }

    // -------- 1) load YOLO runtime from primary CDN, then app --------
    (function start() {
      const PRIMARY = 'https://cdn.jsdelivr.net/gh/vladmandic/yolo/dist/yolo.min.js';
      const MIRROR  = 'https://unpkg.com/@vladmandic/yolo@latest/dist/yolo.min.js';

      function startApp() {
        bootLog('✅ YOLO runtime loaded. Starting app…');
        // IMPORTANT: load app ONLY AFTER YOLO finished loading
        loadScript('script.js', { async: false, onload: () => {
          bootLog('App script loaded.');
          // hide the boot box after a moment; app has HUD of its own
          setTimeout(() => bootBox.remove(), 1200);
        }});
      }

      function tryMirror() {
        bootLog('Primary CDN failed. Trying mirror…');
        loadScript(MIRROR, {
          async: true,
          onload: startApp,
          onerror: () => {
            bootLog('❌ Failed to load YOLO runtime from all sources.');
            alert('Failed to load YOLO runtime. Check your network or try again.');
          }
        });
      }

      bootLog('Loading YOLO runtime…');
      loadScript(PRIMARY, { async: true, onload: startApp, onerror: tryMirror });
    })();
  </script>

  <noscript>
    <div style="color:#fff;background:#222;padding:1em">
      ⚠️ Enable JavaScript to use Cyberpunk V-Scanner.
    </div>
  </noscript>
</body>
</html>

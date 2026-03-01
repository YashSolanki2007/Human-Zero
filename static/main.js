// ============================================================
// HumanZero — main.js
// ============================================================

(function () {
  'use strict';

  // ── Nav scroll effect ────────────────────────────────────
  const nav = document.getElementById('nav');
  window.addEventListener('scroll', () => {
    nav.style.background = window.scrollY > 40
      ? 'rgba(8,12,16,.97)'
      : 'rgba(8,12,16,.82)';
  }, { passive: true });

  // ── Mobile burger ────────────────────────────────────────
  const burger = document.getElementById('navBurger');
  if (burger) {
    burger.addEventListener('click', () => {
      document.querySelector('.nav-links')?.classList.toggle('open');
    });
  }

  // ── Smooth scroll for anchor links ───────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Pipeline step cycling ─────────────────────────────────
  const featureItems = document.querySelectorAll('.feature-item');
  const pipelineSteps = document.querySelectorAll('.pipeline-step');

  featureItems.forEach(item => {
    item.addEventListener('click', () => {
      const idx = item.dataset.idx;
      pipelineSteps.forEach(s => s.classList.remove('active'));
      const target = document.getElementById('ps' + idx);
      if (target) target.classList.add('active');
    });
  });

  let pipelineIdx = 0;
  setInterval(() => {
    pipelineIdx = (pipelineIdx + 1) % pipelineSteps.length;
    pipelineSteps.forEach(s => s.classList.remove('active'));
    if (pipelineSteps[pipelineIdx]) pipelineSteps[pipelineIdx].classList.add('active');
  }, 2800);

  // ── Tab switching (detect section) ────────────────────────
  const tabs = document.querySelectorAll('.tab');
  const uploadHint = document.getElementById('uploadHint');
  let currentTab = 'photo';

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentTab = tab.dataset.tab;
      if (uploadHint) {
        uploadHint.textContent = currentTab === 'photo'
          ? 'JPG, PNG, WebP — max 10 MB'
          : 'MP4, MOV, AVI, MKV, WebM — max 200 MB';
      }
      const fi = document.getElementById('fileInput');
      if (fi) {
        fi.accept = currentTab === 'photo'
          ? '.jpg,.jpeg,.png,.webp'
          : '.mp4,.mov,.avi,.mkv,.webm';
      }
      resetUI();
    });
  });

  // ── Detect upload zone ────────────────────────────────────
  const uploadZone   = document.getElementById('uploadZone');
  const fileInput    = document.getElementById('fileInput');
  const resultPanel  = document.getElementById('resultPanel');
  const loadingPanel = document.getElementById('loadingPanel');
  const errorMsg     = document.getElementById('errorMsg');
  const resetBtn     = document.getElementById('resetBtn');

  if (uploadZone) {
    uploadZone.addEventListener('click', () => fileInput && fileInput.click());
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('dragover');
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });
  }
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });
  }
  if (resetBtn) resetBtn.addEventListener('click', resetUI);

  function resetUI() {
    if (resultPanel)  resultPanel.style.display  = 'none';
    if (loadingPanel) loadingPanel.style.display = 'none';
    if (errorMsg)     errorMsg.style.display     = 'none';
    if (uploadZone)   uploadZone.style.display   = 'flex';
    if (fileInput)    fileInput.value = '';
  }

  function showError(msg) {
    if (errorMsg)     { errorMsg.textContent = msg; errorMsg.style.display = 'block'; }
    if (loadingPanel) loadingPanel.style.display = 'none';
    if (uploadZone)   uploadZone.style.display   = 'flex';
  }

  async function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const photoExts = ['jpg','jpeg','png','webp'];
    const videoExts = ['mp4','mov','avi','mkv','webm','flv','wmv'];
    if (![...photoExts, ...videoExts].includes(ext)) {
      showError('Unsupported file type: .' + ext);
      return;
    }
    if (uploadZone)   uploadZone.style.display   = 'none';
    if (resultPanel)  resultPanel.style.display  = 'none';
    if (errorMsg)     errorMsg.style.display     = 'none';
    if (loadingPanel) loadingPanel.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);
    try {
      const res  = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok || data.error) { showError(data.error || 'An error occurred.'); return; }
      renderResult(data);
    } catch {
      showError('Network error — make sure the server is running.');
    }
  }

  function renderResult(data) {
    if (loadingPanel) loadingPanel.style.display = 'none';
    if (resultPanel)  resultPanel.style.display  = 'block';

    const verdict  = document.getElementById('resultVerdict');
    const badge    = document.getElementById('resultBadge');
    const scoreVal = document.getElementById('scoreValue');
    const barFill  = document.getElementById('scoreBarFill');
    const frameSec = document.getElementById('frameSection');

    const isAI = data.prediction === 1;
    const score = data.score;
    const conf  = (data.confidence || 'Low').toLowerCase();

    if (verdict) {
      verdict.textContent = isAI ? 'AI-Generated' : 'Real';
      verdict.className   = 'result-verdict ' + (isAI ? 'ai' : 'real');
    }
    if (badge) {
      const badgeClass = isAI
        ? (conf === 'high' ? 'high' : conf === 'medium' ? 'medium' : 'low')
        : 'real-high';
      badge.textContent = conf.charAt(0).toUpperCase() + conf.slice(1) + ' confidence';
      badge.className   = 'result-badge ' + badgeClass;
    }
    if (scoreVal) scoreVal.textContent = (score * 100).toFixed(1) + '%';
    if (barFill) {
      barFill.style.width = '0%';
      setTimeout(() => { barFill.style.width = (score * 100) + '%'; }, 50);
    }
    if (frameSec) frameSec.style.display = 'none';
  }

  // ── Mailing list ──────────────────────────────────────────
  const mailForm   = document.getElementById('mailForm');
  const mailThanks = document.getElementById('mailThanks');
  if (mailForm) {
    mailForm.addEventListener('submit', e => {
      e.preventDefault();
      mailForm.style.display = 'none';
      if (mailThanks) mailThanks.style.display = 'block';
    });
  }

  // ── Intersection observer fade-in ─────────────────────────
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animationPlayState = 'running';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.trio-card, .team-card, .feature-item').forEach(el => {
    el.style.animationPlayState = 'paused';
    observer.observe(el);
  });

  // ── FGSM Purify section ───────────────────────────────────
  const purifyZone     = document.getElementById('purifyZone');
  const purifyInput    = document.getElementById('purifyInput');
  const epsilonPanel   = document.getElementById('epsilonPanel');
  const epsilonSlider  = document.getElementById('epsilonSlider');
  const epsilonDisplay = document.getElementById('epsilonDisplay');
  const purifyBtn      = document.getElementById('purifyBtn');
  const purifyReset    = document.getElementById('purifyReset');
  const purifyAgain    = document.getElementById('purifyAgain');
  const purifyLoading  = document.getElementById('purifyLoading');
  const purifyResult   = document.getElementById('purifyResult');
  const purifyError    = document.getElementById('purifyError');
  const purifyDownload = document.getElementById('purifyDownload');

  let purifyFile = null;
  let purifyB64  = null;

  if (purifyZone) {
    purifyZone.addEventListener('click', () => purifyInput && purifyInput.click());
    purifyZone.addEventListener('dragover', e => { e.preventDefault(); purifyZone.classList.add('dragover'); });
    purifyZone.addEventListener('dragleave', () => purifyZone.classList.remove('dragover'));
    purifyZone.addEventListener('drop', e => {
      e.preventDefault();
      purifyZone.classList.remove('dragover');
      if (e.dataTransfer.files[0]) setPurifyFile(e.dataTransfer.files[0]);
    });
  }
  if (purifyInput) {
    purifyInput.addEventListener('change', () => {
      if (purifyInput.files[0]) setPurifyFile(purifyInput.files[0]);
    });
  }

  function setPurifyFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['jpg','jpeg','png','webp'].includes(ext)) {
      showPurifyError('Images only (JPG, PNG, WebP)');
      return;
    }
    purifyFile = file;
    // Preview in zone
    const reader = new FileReader();
    reader.onload = e => {
      const inner = purifyZone ? purifyZone.querySelector('.upload-inner') : null;
      if (inner) {
        inner.innerHTML =
          '<img src="' + e.target.result + '" style="max-height:160px;border-radius:8px;margin-bottom:.8rem;" />' +
          '<p class="upload-title" style="font-size:.85rem">' + file.name + '</p>';
      }
    };
    reader.readAsDataURL(file);
    if (epsilonPanel)  epsilonPanel.style.display  = 'block';
    if (purifyResult)  purifyResult.style.display  = 'none';
    if (purifyError)   purifyError.style.display   = 'none';
  }

  // Slider live update
  if (epsilonSlider) {
    epsilonSlider.addEventListener('input', () => {
      if (epsilonDisplay) epsilonDisplay.textContent = parseFloat(epsilonSlider.value).toFixed(2);
    });
  }

  // Reset / upload different
  if (purifyReset) purifyReset.addEventListener('click', resetPurify);

  // Try again — keep file, just re-show slider
  if (purifyAgain) {
    purifyAgain.addEventListener('click', () => {
      if (purifyResult)  purifyResult.style.display  = 'none';
      if (epsilonPanel)  epsilonPanel.style.display  = 'block';
      if (purifyError)   purifyError.style.display   = 'none';
    });
  }

  function resetPurify() {
    purifyFile = null;
    purifyB64  = null;
    if (purifyInput)   purifyInput.value = '';
    if (epsilonPanel)  epsilonPanel.style.display  = 'none';
    if (purifyResult)  purifyResult.style.display  = 'none';
    if (purifyLoading) purifyLoading.style.display = 'none';
    if (purifyError)   purifyError.style.display   = 'none';
    const inner = purifyZone ? purifyZone.querySelector('.upload-inner') : null;
    if (inner) {
      inner.innerHTML =
        '<div class="upload-icon">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
            '<path d="M9.53 16.122a3 3 0 0 0-5.78 1.128 2.25 2.25 0 0 1-2.4 2.245 4.5 4.5 0 0 0 8.4-2.245c0-.399-.078-.78-.22-1.128Zm0 0a15.998 15.998 0 0 0 3.388-1.62m-5.043-.025a15.994 15.994 0 0 1 1.622-3.395m3.42 3.42a15.995 15.995 0 0 0 4.764-4.648l3.876-5.814a1.151 1.151 0 0 0-1.597-1.597L14.146 6.32a15.996 15.996 0 0 0-4.649 4.763m3.42 3.42a6.776 6.776 0 0 0-3.42-3.42"/>' +
          '</svg>' +
        '</div>' +
        '<p class="upload-title">Drag &amp; drop or click to upload</p>' +
        '<p class="upload-hint">JPG, PNG, WebP \u2014 images only</p>';
    }
  }

  function showPurifyError(msg) {
    if (purifyError)   { purifyError.textContent = msg; purifyError.style.display = 'block'; }
    if (purifyLoading) purifyLoading.style.display = 'none';
    if (epsilonPanel)  epsilonPanel.style.display  = 'block';
  }

  // Generate button
  if (purifyBtn) {
    purifyBtn.addEventListener('click', async () => {
      if (!purifyFile) { showPurifyError('Please upload an image first.'); return; }

      const epsilon = parseFloat(epsilonSlider ? epsilonSlider.value : '0.05');

      if (epsilonPanel)  epsilonPanel.style.display  = 'none';
      if (purifyResult)  purifyResult.style.display  = 'none';
      if (purifyError)   purifyError.style.display   = 'none';
      if (purifyLoading) purifyLoading.style.display = 'block';

      const formData = new FormData();
      formData.append('file', purifyFile);
      formData.append('epsilon', epsilon.toFixed(2));

      try {
        const res  = await fetch('/purify', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok || data.error) { showPurifyError(data.error || 'Server error.'); return; }
        if (purifyLoading) purifyLoading.style.display = 'none';
        renderPurifyResult(data, epsilon);
      } catch {
        showPurifyError('Network error — make sure the server is running.');
      }
    });
  }

  function renderPurifyResult(data, epsilon) {
    purifyB64 = data.image_b64;

    // Original image from local file
    const reader = new FileReader();
    reader.onload = e => {
      const origImg = document.getElementById('purifyOrigImg');
      if (origImg) origImg.src = e.target.result;
    };
    reader.readAsDataURL(purifyFile);

    // Purified image from server
    const advImg = document.getElementById('purifyAdvImg');
    if (advImg) advImg.src = 'data:image/png;base64,' + data.image_b64;

    // Stats under each image
    const origStat = document.getElementById('purifyOrigStat');
    const advStat  = document.getElementById('purifyAdvStat');
    if (origStat) {
      origStat.textContent = data.orig_label + ' \u00b7 ' + (data.orig_ai_prob * 100).toFixed(1) + '% AI';
      origStat.className   = 'purify-img-stat ' + (data.orig_label === 'AI-Generated' ? 'ai' : 'real');
    }
    if (advStat) {
      advStat.textContent = data.adv_label + ' \u00b7 ' + (data.adv_ai_prob * 100).toFixed(1) + '% AI';
      advStat.className   = 'purify-img-stat ' + (data.adv_label === 'AI-Generated' ? 'ai' : 'real');
    }

    // Epsilon label
    const epsLabel = document.getElementById('purifyEpsLabel');
    if (epsLabel) epsLabel.textContent = '\u03b5 = ' + epsilon.toFixed(2);

    // Flip banner
    const flipEl = document.getElementById('purifyFlip');
    if (flipEl) {
      if (data.flipped) {
        flipEl.textContent = '\u2713 Prediction flipped: ' + data.orig_label + ' \u2192 ' + data.adv_label;
        flipEl.className   = 'purify-flip success';
      } else {
        flipEl.textContent = 'Prediction unchanged (' + data.adv_label + ') \u2014 try a higher \u03b5';
        flipEl.className   = 'purify-flip fail';
      }
    }

    if (purifyResult) purifyResult.style.display = 'block';
  }

  // Download purified image
  if (purifyDownload) {
    purifyDownload.addEventListener('click', () => {
      if (!purifyB64) return;
      const a = document.createElement('a');
      a.href     = 'data:image/png;base64,' + purifyB64;
      a.download = 'humanzero_purified.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });
  }

})();
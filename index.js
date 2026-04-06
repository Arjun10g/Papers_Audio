(() => {
  const BACKS   = [
    "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1920&q=80",  // Splines
    "https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1920&q=80",  // ML
    "https://images.unsplash.com/photo-1545987796-200677ee1011?w=1920&q=80",     // Non-Linearity
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1920&q=80",     // Longitudinal
    "https://images.unsplash.com/photo-1509228468518-180dd4864904?w=1920&q=80",  // Spline Interpretation
    "https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=1920&q=80",  // Trees
    "https://images.unsplash.com/photo-1558618666-fcd25c85f82e?w=1920&q=80",     // SVM
    "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1920&q=80",  // Neural Networks
    "https://images.unsplash.com/photo-1478720568477-152d9b164e26?w=1920&q=80",  // Rashomon Effect
    "https://images.unsplash.com/photo-1494232410401-ad00d5433cfa?w=1920&q=80",  // Rashomon Analysis
    "https://images.unsplash.com/photo-1527474305487-b87b222841cc?w=1920&q=80",  // Boosting/Ensemble
  ];
  const BG_BASE = "";

  const $  = s => document.querySelector(s);
  const $$ = s => Array.from(document.querySelectorAll(s));
  const pad = n => String(n).padStart(2, '0');
  const fmt = s => !isFinite(s) ? "00:00" : `${pad(Math.floor(s / 60))}:${pad(Math.floor(s % 60))}`;

  const audio    = $('#player');
  const plToggle = $('#pl-toggle');
  const plBar    = $('#pl-bar');
  const plFill   = $('#pl-fill');
  const plCur    = $('#pl-cur');
  const plDur    = $('#pl-dur');
  const plNum    = $('#pl-num');
  const plName   = $('#pl-name');
  const plRate   = $('#pl-rate');
  const plVol    = $('#pl-volume');
  const plTheme  = $('#pl-theme');
  const player   = $('.player');
  const rightPane = $('.right');
  const btnPrev  = $('#pl-prev');
  const btnNext  = $('#pl-next');

  const chapterLinks = $$('.chapters a');
  const PLAYLIST = chapterLinks.map((a, i) => ({
    src: a.dataset.src,
    title: a.querySelector('.ch-title')?.textContent.trim() || `Chapter ${i + 1}`
  }));

  if (!PLAYLIST.length) return;

  let index = 0;

  audio.playbackRate = parseFloat(plRate.value || "1");
  audio.volume = parseFloat(plVol.value);

  // ── Make player draggable via GSAP ──
  if (typeof Draggable !== 'undefined') {
    Draggable.create(player, {
      type: "x,y",
      bounds: "body",
      edgeResistance: 0.65,
      inertia: true,
      cursor: "grab",
      activeCursor: "grabbing",
      onClick: function() {},
      allowEventDefault: true,
    });
  }

  function setBackgroundForIndex(i) {
    const bg = BACKS[i] || BACKS[0];
    rightPane.style.backgroundImage = `url('${BG_BASE}${bg}')`;
    rightPane.style.backgroundSize = 'cover';
    rightPane.style.backgroundRepeat = 'no-repeat';
    rightPane.style.backgroundPosition = 'center';
  }

  function highlightActive(i) {
    chapterLinks.forEach(el => el.classList.remove('active'));
    chapterLinks[i]?.classList.add('active');
  }

  function load(k, autoplay = false) {
    index = (k + PLAYLIST.length) % PLAYLIST.length;
    const tr = PLAYLIST[index];

    plNum.textContent = String(index + 1);
    plName.textContent = tr.title;
    setBackgroundForIndex(index);
    highlightActive(index);

    audio.pause();
    audio.src = tr.src;
    audio.load();

    const tryPlay = () => {
      if (!autoplay) { plToggle.textContent = '\u25B6'; return; }
      audio.play().then(() => {
        plToggle.textContent = '\u23F8';
      }).catch(() => {
        plToggle.textContent = '\u25B6';
      });
    };

    if (audio.readyState >= 2) {
      tryPlay();
    } else {
      audio.addEventListener('canplay', tryPlay, { once: true });
    }
  }

  // ── Events ──
  chapterLinks.forEach((link, idx) => {
    link.addEventListener('click', e => { e.preventDefault(); load(idx, true); });
  });

  btnPrev?.addEventListener('click', () => load(index - 1, true));
  btnNext?.addEventListener('click', () => load(index + 1, true));
  plToggle.addEventListener('click', () => audio.paused ? audio.play() : audio.pause());
  plRate.addEventListener('change', () => { audio.playbackRate = parseFloat(plRate.value || "1"); });
  plVol.addEventListener('input', () => { audio.volume = parseFloat(plVol.value); });

  plBar.addEventListener('click', e => {
    const r = plBar.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    if (isFinite(audio.duration)) audio.currentTime = t * audio.duration;
  });

  audio.addEventListener('loadedmetadata', () => { plDur.textContent = fmt(audio.duration); });
  audio.addEventListener('timeupdate', () => {
    if (!isFinite(audio.duration)) return;
    plCur.textContent = fmt(audio.currentTime);
    plFill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
  });

  audio.addEventListener('play', () => { plToggle.textContent = '\u23F8'; });
  audio.addEventListener('pause', () => { plToggle.textContent = '\u25B6'; });
  audio.addEventListener('ended', () => load(index + 1, true));

  plTheme.addEventListener('click', () => {
    const cur = player.getAttribute('data-theme') || 'dark';
    player.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  });

  audio.addEventListener('error', () => {
    const e = audio.error;
    const codes = { 1: 'ABORTED', 2: 'NETWORK', 3: 'DECODE', 4: 'SRC_NOT_SUPPORTED' };
    console.error('Audio error:', codes[e?.code], 'src =', audio.currentSrc);
  });

  load(0, false);
})();

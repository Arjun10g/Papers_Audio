(() => {
  // ========= CONFIG =========
  // Backgrounds per index (0 -> back.png, 1 -> back2.png, etc.)
  const BACKS   = ["back.png", "back2.png", "back3.png"]; // add more or fewer as desired
  const BG_BASE = ""; // e.g., "img/" if your images live in /img
  // =========================

  const $  = s => document.querySelector(s);
  const $$ = s => Array.from(document.querySelectorAll(s));
  const pad = n => String(n).padStart(2, '0');
  const fmt = s => !isFinite(s) ? "00:00" : `${pad(Math.floor(s/60))}:${pad(Math.floor(s%60))}`;

  // Core elements
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
  const player   = document.querySelector('.player');
  const rightPane= document.querySelector('.right');
  const btnPrev  = $('#pl-prev');
  const btnNext  = $('#pl-next');

  // Build PLAYLIST from the left list (so it matches your HTML)
  const chapterLinks = $$('.chapters a');
  const PLAYLIST = chapterLinks.map((a, i) => ({
    src: a.dataset.src,                           // e.g. "audio/splines.mp3"
    title: a.querySelector('.ch-title')?.textContent.trim() || `Chapter ${i+1}`
  }));

  // Guard: if no chapters found, bail
  if (!PLAYLIST.length) {
    console.warn('No chapters found. Make sure .chapters a elements exist.');
    return;
  }

  let index = 0;

  // initial audio prefs from UI
  audio.playbackRate = parseFloat(plRate.value || "1");
  audio.volume       = parseFloat(plVol.value);

  function setBackgroundForIndex(i) {
    const bg = BACKS[i] || BACKS[0];
    rightPane.style.backgroundImage = `url('${BG_BASE}${bg}')`;
    rightPane.style.backgroundSize = 'contain';
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

    // UI labels
    plNum.textContent = String(index + 1);
    plName.textContent = tr.title;

    // background by index
    setBackgroundForIndex(index);
    // highlight in list
    highlightActive(index);

    // load audio
    audio.pause();
    audio.src = tr.src; // uses href from chapter list
    audio.load();

    const tryPlay = () => {
      if (!autoplay) { plToggle.textContent = '▶'; return; }
      audio.play().then(() => {
        plToggle.textContent = '⏸';
      }).catch(err => {
        // autoplay may be blocked until user gesture
        console.warn('play() blocked/failed:', err);
        plToggle.textContent = '▶';
      });
    };

    if (audio.readyState >= 2) {
      tryPlay();
    } else {
      audio.addEventListener('canplay', tryPlay, { once: true });
    }
  }

  // Clicking a chapter loads & plays it
  chapterLinks.forEach((link, idx) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      load(idx, true);
    });
  });

  // Prev/Next
  btnPrev?.addEventListener('click', () => load(index - 1, true));
  btnNext?.addEventListener('click', () => load(index + 1, true));

  // Play/Pause
  plToggle.addEventListener('click', () => audio.paused ? audio.play() : audio.pause());

  // Rate & Volume
  plRate.addEventListener('change', () => { audio.playbackRate = parseFloat(plRate.value || "1"); });
  plVol .addEventListener('input',  () => { audio.volume       = parseFloat(plVol.value); });

  // Seek
  plBar.addEventListener('click', (e) => {
    const r = plBar.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    if (isFinite(audio.duration)) audio.currentTime = t * audio.duration;
  });

  // Time/progress updates
  audio.addEventListener('loadedmetadata', () => { plDur.textContent = fmt(audio.duration); });
  audio.addEventListener('timeupdate', () => {
    if (!isFinite(audio.duration)) return;
    plCur.textContent = fmt(audio.currentTime);
    plFill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
  });

  audio.addEventListener('play',  () => { plToggle.textContent = '⏸'; });
  audio.addEventListener('pause', () => { plToggle.textContent = '▶'; });

  // Auto-advance
  audio.addEventListener('ended', () => load(index + 1, true));

  // Theme toggle
  plTheme.addEventListener('click', () => {
    const cur = player.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    player.setAttribute('data-theme', next);
  });

  // Optional: basic error logging (helps catch 404/MIME issues)
  audio.addEventListener('error', () => {
    const e = audio.error;
    const codes = {1:'ABORTED', 2:'NETWORK', 3:'DECODE', 4:'SRC_NOT_SUPPORTED'};
    console.error('Audio error:', codes[e?.code], 'src =', audio.currentSrc);
  });

  // Initial load (no autoplay)
  load(0, false);
})();
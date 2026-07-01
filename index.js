(() => {
  // ── Background images per chapter (by index) ──
  const BACKS = [
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
    "https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1920&q=80",  // Decomposed Tree
    "https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=1920&q=80",  // Transformers/LLMs
    "https://images.unsplash.com/photo-1639322537228-f710d846310a?w=1920&q=80",  // Embedders
    "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=1920&q=80",  // Recurrent Depth
    "https://images.unsplash.com/photo-1591405351990-4726e331f141?w=1920&q=80",  // Hardware Ouro
  ];
  const BG_BASE = "";

  const SKIP = 15;                       // seconds for skip buttons / arrow keys
  const LS = "audible.";                 // localStorage namespace

  const $  = s => document.querySelector(s);
  const $$ = s => Array.from(document.querySelectorAll(s));
  const pad = n => String(n).padStart(2, '0');
  const fmt = s => !isFinite(s) ? "00:00" : `${pad(Math.floor(s / 60))}:${pad(Math.floor(s % 60))}`;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const store = {
    get: (k, d) => { try { const v = localStorage.getItem(LS + k); return v === null ? d : v; } catch { return d; } },
    set: (k, v) => { try { localStorage.setItem(LS + k, v); } catch {} },
  };

  // ── Elements ──
  const audio    = $('#player');
  const plToggle = $('#pl-toggle');
  const plBar    = $('#pl-bar');
  const plFill   = $('#pl-fill');
  const plBuf    = $('#pl-buffered');
  const plCur    = $('#pl-cur');
  const plDur    = $('#pl-dur');
  const plRemain = $('#pl-remain');
  const plNum    = $('#pl-num');
  const plName   = $('#pl-name');
  const plRate   = $('#pl-rate');
  const plVol    = $('#pl-volume');
  const plTheme  = $('#pl-theme');
  const plSleep  = $('#pl-sleep');
  const plRead   = $('#pl-read');
  const plHelp   = $('#pl-help');
  const player   = $('.player');
  const rightPane = $('.right');
  const btnPrev  = $('#pl-prev');
  const btnNext  = $('#pl-next');
  const btnBack  = $('#pl-back');
  const btnFwd   = $('#pl-fwd');

  const search   = $('#ch-search');
  const searchCount = $('#ch-search-count');

  const reader     = $('#reader');
  const readerBody = $('#reader-body');
  const readerTitle = $('#reader-title');
  const readerClose = $('#reader-close');
  const readerFontUp = $('#reader-font-up');
  const readerFontDn = $('#reader-font-dn');

  const helpOverlay = $('#help-overlay');
  const helpClose   = $('#help-close');

  const chapterLinks = $$('.chapters a');
  const PLAYLIST = chapterLinks.map((a, i) => ({
    src: a.dataset.src || "",
    doc: a.dataset.doc || "",
    title: a.querySelector('.ch-title')?.textContent.trim() || `Chapter ${i + 1}`,
    li: a.closest('li'),
  }));

  if (!PLAYLIST.length) return;

  let index = 0;
  let lastVolume = parseFloat(store.get('volume', '1')) || 1;
  let sleepTimer = null;
  let sleepAtEnd = false;
  const docCache = new Map();

  // ── Restore persisted settings ──
  plRate.value = store.get('rate', plRate.value);
  audio.playbackRate = parseFloat(plRate.value || "1");
  plVol.value = lastVolume;
  audio.volume = lastVolume;
  player.setAttribute('data-theme', store.get('theme', 'dark'));
  document.body.setAttribute('data-theme', store.get('theme', 'dark'));
  let readerFont = parseFloat(store.get('readerFont', '1.05')) || 1.05;
  readerBody.style.setProperty('--reader-font', readerFont + 'rem');

  // ── Draggable player ──
  if (typeof Draggable !== 'undefined') {
    Draggable.create(player, {
      type: "x,y", bounds: "body", edgeResistance: 0.65, inertia: true,
      cursor: "grab", activeCursor: "grabbing", allowEventDefault: true,
    });
  }

  // ── Position persistence ──
  const posKey = src => 'pos.' + src;
  function savePosition() {
    const tr = PLAYLIST[index];
    if (tr.src && isFinite(audio.currentTime) && audio.currentTime > 5 &&
        audio.duration && audio.currentTime < audio.duration - 5) {
      store.set(posKey(tr.src), String(audio.currentTime));
    }
    store.set('lastIndex', String(index));
  }
  setInterval(() => { if (!audio.paused) savePosition(); }, 5000);
  window.addEventListener('beforeunload', savePosition);
  audio.addEventListener('pause', savePosition);

  // ── UI helpers ──
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

  function setTransportEnabled(on) {
    [plToggle, btnBack, btnFwd].forEach(b => { b.disabled = !on; });
    plBar.style.pointerEvents = on ? 'auto' : 'none';
    plBar.style.opacity = on ? '1' : '0.5';
  }

  function updateMediaSession(tr) {
    if (!('mediaSession' in navigator)) return;
    try {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: tr.title,
        artist: 'Papers, as Audio',
        album: `Chapter ${index + 1}`,
        artwork: [{ src: BACKS[index] || BACKS[0], sizes: '512x512', type: 'image/jpeg' }],
      });
    } catch {}
  }

  // ── Load a chapter ──
  function load(k, autoplay = false, openReader = false) {
    index = (k + PLAYLIST.length) % PLAYLIST.length;
    const tr = PLAYLIST[index];

    plNum.textContent = String(index + 1);
    plName.textContent = tr.title;
    setBackgroundForIndex(index);
    highlightActive(index);
    plRead.disabled = !tr.doc;
    plRead.classList.toggle('is-active', reader.dataset.open === 'true' && !!tr.doc);
    updateMediaSession(tr);
    store.set('lastIndex', String(index));

    plFill.style.width = '0%';
    plBuf.style.width = '0%';

    if (!tr.src) {
      // Reading-only chapter (audio not generated yet)
      audio.pause();
      audio.removeAttribute('src');
      audio.load();
      setTransportEnabled(false);
      plToggle.textContent = '▶';
      plCur.textContent = '—:—';
      plDur.textContent = 'read';
      plRemain.textContent = '';
      if (tr.doc) openDoc(tr);
      return;
    }

    setTransportEnabled(true);
    audio.pause();
    audio.src = tr.src;
    audio.load();

    const resume = parseFloat(store.get(posKey(tr.src), '0')) || 0;

    const onReady = () => {
      if (resume > 1 && isFinite(audio.duration) && resume < audio.duration - 5) {
        audio.currentTime = resume;
      }
      if (autoplay) {
        audio.play().then(() => { plToggle.textContent = '⏸'; })
                    .catch(() => { plToggle.textContent = '▶'; });
      } else {
        plToggle.textContent = '▶';
      }
    };

    if (audio.readyState >= 1) onReady();
    else audio.addEventListener('loadedmetadata', onReady, { once: true });

    if (openReader && tr.doc) openDoc(tr);
    else if (reader.dataset.open === 'true' && tr.doc) openDoc(tr); // keep reader synced
  }

  // ── Reading pane ──
  async function fetchDoc(path) {
    if (docCache.has(path)) return docCache.get(path);
    const res = await fetch(path);
    if (!res.ok) throw new Error(res.status);
    const text = await res.text();
    docCache.set(path, text);
    return text;
  }

  function renderMarkdown(text) {
    if (window.marked) {
      return (typeof marked.parse === 'function') ? marked.parse(text) : marked(text);
    }
    // Fallback: minimal escaping in a <pre>
    const esc = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return `<pre>${esc}</pre>`;
  }

  async function openDoc(tr) {
    readerTitle.textContent = tr.title;
    reader.dataset.open = 'true';
    reader.setAttribute('aria-hidden', 'false');
    plRead.classList.add('is-active');
    if (!tr.doc) {
      readerBody.innerHTML = '<p class="reader-empty">No transcript available for this chapter yet.</p>';
      return;
    }
    readerBody.innerHTML = '<p class="reader-empty">Loading transcript…</p>';
    try {
      const text = await fetchDoc(tr.doc);
      readerBody.innerHTML = renderMarkdown(text);
      readerBody.scrollTop = 0;
      if (window.MathJax && MathJax.Hub) MathJax.Hub.Queue(["Typeset", MathJax.Hub, readerBody]);
    } catch (e) {
      readerBody.innerHTML = `<p class="reader-empty">Could not load transcript (${e.message}). ` +
        `If you opened this file directly, serve the folder over HTTP so the transcript can be fetched.</p>`;
    }
  }

  function closeReader() {
    reader.dataset.open = 'false';
    reader.setAttribute('aria-hidden', 'true');
    plRead.classList.remove('is-active');
  }

  function toggleReader() {
    if (reader.dataset.open === 'true') closeReader();
    else openDoc(PLAYLIST[index]);
  }

  plRead.addEventListener('click', toggleReader);
  readerClose.addEventListener('click', closeReader);
  readerFontUp.addEventListener('click', () => setReaderFont(readerFont + 0.1));
  readerFontDn.addEventListener('click', () => setReaderFont(readerFont - 0.1));

  function setReaderFont(v) {
    readerFont = clamp(v, 0.8, 1.8);
    readerBody.style.setProperty('--reader-font', readerFont.toFixed(2) + 'rem');
    store.set('readerFont', readerFont.toFixed(2));
  }

  // ── Sleep timer ──
  function clearSleep() {
    if (sleepTimer) { clearTimeout(sleepTimer); sleepTimer = null; }
    sleepAtEnd = false;
    plSleep.classList.remove('is-active');
  }

  plSleep.addEventListener('change', () => {
    clearSleep();
    const v = plSleep.value;
    if (v === '0') return;
    plSleep.classList.add('is-active');
    if (v === 'end') { sleepAtEnd = true; return; }
    const secs = parseInt(v, 10);
    sleepTimer = setTimeout(() => { audio.pause(); clearSleep(); plSleep.value = '0'; }, secs * 1000);
  });

  // ── Transport ──
  function togglePlay() {
    if (!PLAYLIST[index].src) return;
    audio.paused ? audio.play() : audio.pause();
  }
  function skip(delta) {
    if (!isFinite(audio.duration)) return;
    audio.currentTime = clamp(audio.currentTime + delta, 0, audio.duration);
  }

  chapterLinks.forEach((link, idx) => {
    link.addEventListener('click', e => {
      e.preventDefault();
      const hasAudio = !!PLAYLIST[idx].src;
      load(idx, hasAudio, !hasAudio); // reading-only chapters open the reader
    });
  });

  btnPrev.addEventListener('click', () => load(index - 1, true));
  btnNext.addEventListener('click', () => load(index + 1, true));
  btnBack.addEventListener('click', () => skip(-SKIP));
  btnFwd.addEventListener('click', () => skip(SKIP));
  plToggle.addEventListener('click', togglePlay);

  plRate.addEventListener('change', () => {
    audio.playbackRate = parseFloat(plRate.value || "1");
    store.set('rate', plRate.value);
  });
  plVol.addEventListener('input', () => {
    audio.volume = parseFloat(plVol.value);
    lastVolume = audio.volume || lastVolume;
    store.set('volume', String(audio.volume));
  });

  plBar.addEventListener('click', e => {
    const r = plBar.getBoundingClientRect();
    const t = clamp((e.clientX - r.left) / r.width, 0, 1);
    if (isFinite(audio.duration)) audio.currentTime = t * audio.duration;
  });

  // ── Audio events ──
  audio.addEventListener('loadedmetadata', () => { plDur.textContent = fmt(audio.duration); });
  audio.addEventListener('timeupdate', () => {
    if (!isFinite(audio.duration)) return;
    plCur.textContent = fmt(audio.currentTime);
    plRemain.textContent = '-' + fmt(audio.duration - audio.currentTime);
    plFill.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
    if ('mediaSession' in navigator && navigator.mediaSession.setPositionState) {
      try { navigator.mediaSession.setPositionState({
        duration: audio.duration, position: audio.currentTime, playbackRate: audio.playbackRate,
      }); } catch {}
    }
  });
  audio.addEventListener('progress', () => {
    try {
      if (audio.buffered.length && isFinite(audio.duration)) {
        const end = audio.buffered.end(audio.buffered.length - 1);
        plBuf.style.width = `${(end / audio.duration) * 100}%`;
      }
    } catch {}
  });
  audio.addEventListener('play',  () => { plToggle.textContent = '⏸'; });
  audio.addEventListener('pause', () => { plToggle.textContent = '▶'; });
  audio.addEventListener('ended', () => {
    const tr = PLAYLIST[index];
    if (tr.src) store.set(posKey(tr.src), '0'); // finished → reset resume
    if (sleepAtEnd) { clearSleep(); plSleep.value = '0'; return; }
    load(index + 1, true);
  });
  audio.addEventListener('error', () => {
    if (!PLAYLIST[index].src) return;
    const e = audio.error;
    const codes = { 1: 'ABORTED', 2: 'NETWORK', 3: 'DECODE', 4: 'SRC_NOT_SUPPORTED' };
    console.error('Audio error:', codes[e?.code], 'src =', audio.currentSrc);
  });

  // ── Theme ──
  plTheme.addEventListener('click', () => {
    const cur = player.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    player.setAttribute('data-theme', next);
    document.body.setAttribute('data-theme', next);
    store.set('theme', next);
  });

  function toggleMute() {
    if (audio.volume > 0) { lastVolume = audio.volume; audio.volume = 0; plVol.value = 0; }
    else { audio.volume = lastVolume || 1; plVol.value = audio.volume; }
    store.set('volume', String(audio.volume));
  }

  // ── Media Session actions ──
  if ('mediaSession' in navigator) {
    const ms = navigator.mediaSession;
    try {
      ms.setActionHandler('play', () => audio.play());
      ms.setActionHandler('pause', () => audio.pause());
      ms.setActionHandler('previoustrack', () => load(index - 1, true));
      ms.setActionHandler('nexttrack', () => load(index + 1, true));
      ms.setActionHandler('seekbackward', () => skip(-SKIP));
      ms.setActionHandler('seekforward', () => skip(SKIP));
      ms.setActionHandler('seekto', d => { if (d.seekTime != null && isFinite(audio.duration)) audio.currentTime = d.seekTime; });
    } catch {}
  }

  // ── Search / filter ──
  function applyFilter() {
    const q = (search.value || '').trim().toLowerCase();
    let shown = 0;
    PLAYLIST.forEach(tr => {
      const match = !q || tr.title.toLowerCase().includes(q) || String(PLAYLIST.indexOf(tr) + 1) === q;
      tr.li.classList.toggle('is-hidden', !match);
      if (match) shown++;
    });
    searchCount.textContent = q ? `${shown}/${PLAYLIST.length}` : '';
  }
  search.addEventListener('input', applyFilter);

  // ── Help overlay ──
  function toggleHelp(force) {
    const open = force != null ? force : helpOverlay.dataset.open !== 'true';
    helpOverlay.dataset.open = String(open);
    helpOverlay.setAttribute('aria-hidden', String(!open));
  }
  plHelp.addEventListener('click', () => toggleHelp());
  helpClose.addEventListener('click', () => toggleHelp(false));
  helpOverlay.addEventListener('click', e => { if (e.target === helpOverlay) toggleHelp(false); });

  // ── Keyboard shortcuts ──
  document.addEventListener('keydown', e => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (e.key === 'Escape') {
      if (helpOverlay.dataset.open === 'true') return toggleHelp(false);
      if (reader.dataset.open === 'true') return closeReader();
      if (typing) document.activeElement.blur();
      return;
    }
    if (typing) return;
    switch (e.key) {
      case ' ': case 'k': e.preventDefault(); togglePlay(); break;
      case 'ArrowRight': e.preventDefault(); e.shiftKey ? load(index + 1, true) : skip(SKIP); break;
      case 'ArrowLeft':  e.preventDefault(); e.shiftKey ? load(index - 1, true) : skip(-SKIP); break;
      case 'ArrowUp':    e.preventDefault(); plVol.value = clamp(audio.volume + 0.05, 0, 1); plVol.dispatchEvent(new Event('input')); break;
      case 'ArrowDown':  e.preventDefault(); plVol.value = clamp(audio.volume - 0.05, 0, 1); plVol.dispatchEvent(new Event('input')); break;
      case ']': stepRate(1); break;
      case '[': stepRate(-1); break;
      case 'm': case 'M': toggleMute(); break;
      case 'r': case 'R': toggleReader(); break;
      case 't': case 'T': plTheme.click(); break;
      case '/': e.preventDefault(); search.focus(); break;
      case '?': toggleHelp(); break;
      case 'n': case 'N': load(index + 1, true); break;
      case 'p': case 'P': load(index - 1, true); break;
      default:
        if (/^[0-9]$/.test(e.key) && isFinite(audio.duration)) {
          audio.currentTime = (parseInt(e.key, 10) / 10) * audio.duration;
        }
    }
  });

  function stepRate(dir) {
    const opts = Array.from(plRate.options).map(o => o.value);
    let i = opts.indexOf(plRate.value);
    i = clamp(i + dir, 0, opts.length - 1);
    plRate.value = opts[i];
    plRate.dispatchEvent(new Event('change'));
  }

  // ── Boot ──
  applyFilter();
  const startIndex = clamp(parseInt(store.get('lastIndex', '0'), 10) || 0, 0, PLAYLIST.length - 1);
  load(startIndex, false);
})();

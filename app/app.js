/* ==========================================================================
   Papers, as Audio
   ==========================================================================

   library.json (schema 1) — fetched from the Hugging Face dataset at runtime:

   {
     schema: 1, title, generated, repo, revision,
     base: "https://huggingface.co/datasets/<repo>/resolve/<rev>/",
     lectures: [{
       id, n, title, series,
       audio,            // path under base  -> base + audio is the mp3 URL
       bytes, duration,  // bytes = total size, duration = seconds (float)
       doc,              // path under base, or null for audio-only lectures
       bg,               // artwork URL (Unsplash; ?w= can be resized)
       published, sha256,
       chapters?: [      // OPTIONAL in-lecture section markers, ascending
         { title: "Cold open", start: 0 },
         { title: "Section 2 – …", start: 312.4 }
       ]
     }]
   }

   `base` is pinned to a commit, so a full audio URL is immutable and doubles
   as the Cache API key for an offline download.
   ========================================================================== */

(() => {
'use strict';

const CFG = window.APP_CONFIG || {};
const LIBRARY_URL = CFG.LIBRARY_URL || './library.json';
const APP_VERSION = CFG.APP_VERSION || 'dev';

const AUDIO_CACHE = 'audio-v1';
const DOC_CACHE   = 'docs-v1';

const SKIP_BACK = 15;
const SKIP_FWD  = 30;
const RATES     = [0.8, 1, 1.25, 1.5, 1.75, 2];
const SLEEPS    = [
  { v: 'off', label: 'Off' },
  { v: 'end', label: 'End of lecture' },
  { v: 900,   label: '15 minutes' },
  { v: 1800,  label: '30 minutes' },
  { v: 2700,  label: '45 minutes' },
  { v: 3600,  label: '60 minutes' },
];

const MARKED_URL  = 'https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js';
const KATEX_CSS   = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css';
const KATEX_JS    = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js';
const KATEX_AUTO  = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js';

/* ── tiny helpers ───────────────────────────────────────────────────────── */

const $ = (s, r) => (r || document).querySelector(s);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pad2 = n => String(n).padStart(2, '0');

const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** 1:07:24 / 12:34 */
function hms(s) {
  if (!isFinite(s) || s < 0) s = 0;
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${pad2(m)}:${pad2(sec)}` : `${m}:${pad2(sec)}`;
}
/** "46 min" / "1 h 12 min" */
function mins(s) {
  if (!isFinite(s) || s <= 0) return '—';
  const m = Math.round(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${pad2(m % 60)} min` : `${m} min`;
}
function bytesLabel(b) {
  if (!isFinite(b) || b <= 0) return '0 MB';
  if (b < 1024 * 1024) return `${Math.round(b / 1024)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1048576).toFixed(b < 10485760 ? 1 : 0)} MB`;
  return `${(b / 1073741824).toFixed(2)} GB`;
}

const NS = 'pa.';
const store = {
  get(k, d) { try { const v = localStorage.getItem(NS + k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(NS + k, String(v)); return true; } catch (e) { return false; } },
  del(k)    { try { localStorage.removeItem(NS + k); } catch (e) {} },
  getJSON(k, d) { try { const v = localStorage.getItem(NS + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  setJSON(k, v) { try { localStorage.setItem(NS + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

function loadScript(src) {
  return new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) return res();
    const s = document.createElement('script');
    s.src = src; s.async = true; s.crossOrigin = 'anonymous';
    s.onload = () => res();
    s.onerror = () => rej(new Error('Could not load ' + src));
    document.head.appendChild(s);
  });
}
function loadCss(href) {
  return new Promise(res => {
    if (document.querySelector(`link[href="${href}"]`)) return res();
    const l = document.createElement('link');
    l.rel = 'stylesheet'; l.href = href; l.crossOrigin = 'anonymous';
    l.onload = l.onerror = () => res();
    document.head.appendChild(l);
  });
}

/* ── DOM ────────────────────────────────────────────────────────────────── */

const audio = $('#audio');

const el = {
  topbar: $('#topbar'), storageLine: $('#storageLine'), offlinePill: $('#offlinePill'),
  btnInstall: $('#btnInstall'), btnTheme: $('#btnTheme'), btnHelp: $('#btnHelp'),
  search: $('#search'), searchClear: $('#searchClear'), btnDownloadAll: $('#btnDownloadAll'),
  continueSlot: $('#continueSlot'), list: $('#list'), foot: $('#foot'),

  mini: $('#mini'), miniFill: $('#miniFill'), miniOpen: $('#miniOpen'), miniArt: $('#miniArt'),
  miniTitle: $('#miniTitle'), miniSub: $('#miniSub'), miniPlay: $('#miniPlay'), miniFwd: $('#miniFwd'),

  sheet: $('#sheet'), sheetBg: $('#sheetBg'), sheetGrab: $('#sheetGrab'), sheetNum: $('#sheetNum'),
  btnCloseSheet: $('#btnCloseSheet'), cover: $('#cover'), coverWrap: $('.cover-wrap'),
  npSeries: $('#npSeries'), npTitle: $('#npTitle'),
  chapBar: $('#chapBar'), chapNow: $('#chapNow'), btnChapters: $('#btnChapters'),
  btnChapPrev: $('#btnChapPrev'), btnChapNext: $('#btnChapNext'),
  chapPanel: $('#chapPanel'), chapItems: $('#chapItems'), chapClose: $('#chapClose'),
  scrub: $('#scrub'), ticks: $('#ticks'), tCur: $('#tCur'), tTotal: $('#tTotal'), tRemain: $('#tRemain'),
  btnPrev: $('#btnPrev'), btnBack: $('#btnBack'), btnPlay: $('#btnPlay'), btnFwd: $('#btnFwd'), btnNext: $('#btnNext'),
  btnRate: $('#btnRate'), rateVal: $('#rateVal'), btnSleep: $('#btnSleep'), sleepLabel: $('#sleepLabel'),
  btnDownload: $('#btnDownload'), dlIcon: $('#dlIcon'), dlLabel: $('#dlLabel'), btnRead: $('#btnRead'),

  reader: $('#reader'), readerTitle: $('#readerTitle'), readerBody: $('#readerBody'),
  readerClose: $('#readerClose'), fontUp: $('#fontUp'), fontDn: $('#fontDn'),

  picker: $('#picker'), pickerTitle: $('#pickerTitle'), pickerOpts: $('#pickerOpts'),
  help: $('#help'), helpClose: $('#helpClose'),
  toasts: $('#toasts'),
};

const setIcon = (btn, id) => {
  const use = btn && btn.querySelector('use');
  if (use) use.setAttribute('href', '#' + id);
};

/* ── state ──────────────────────────────────────────────────────────────── */

const state = {
  lib: null,
  base: '',
  lectures: [],
  bySeries: [],
  source: 'none',        // network | cache | bundled | none
  i: -1,                 // index of the loaded lecture, -1 = nothing loaded
  rate: parseFloat(store.get('rate', '1')) || 1,
  newIds: new Set(),
  downloaded: new Set(), // full audio URLs present in AUDIO_CACHE
  dl: new Map(),         // id -> { loaded, total, ctrl }
  bulk: false,
  sleep: { mode: 'off', deadline: 0 },
  chapI: -1,
  scrubbing: false,
  pendingSeek: null,
  swReg: null,
  updateAccepted: false,
  installPrompt: null,
};

const current = () => (state.i >= 0 ? state.lectures[state.i] : null);
const fullUrl = p => {
  if (!p) return '';
  if (/^https?:\/\//i.test(p)) return p;
  return state.base + String(p).replace(/^\.?\//, '');
};
const audioUrl = l => (l ? fullUrl(l.audio) : '');

function artUrl(bg, w) {
  if (!bg) return './icons/icon.svg';
  try {
    const u = new URL(bg);
    if (u.searchParams.has('w')) {
      u.searchParams.set('w', String(w));
      u.searchParams.set('q', w > 400 ? '80' : '65');
    }
    return u.href;
  } catch (e) { return bg; }
}

const posKey  = id => 'pos.' + id;
const doneKey = id => 'done.' + id;
const savedPos  = id => parseFloat(store.get(posKey(id), '0')) || 0;
const isDone    = id => store.get(doneKey(id), '') === '1';
function progressOf(l) {
  if (isDone(l.id)) return 1;
  const d = Number(l.duration) || 0;
  if (!d) return 0;
  return clamp(savedPos(l.id) / d, 0, 1);
}

/* ── toasts ─────────────────────────────────────────────────────────────── */

function toast(msg, opts) {
  opts = opts || {};
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `<span>${esc(msg)}</span>`;
  if (opts.action) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = opts.action;
    b.addEventListener('click', () => { kill(); opts.onAction && opts.onAction(); });
    t.appendChild(b);
  }
  el.toasts.appendChild(t);
  let dead = false;
  function kill() {
    if (dead) return; dead = true;
    t.classList.add('is-out');
    setTimeout(() => t.remove(), 260);
  }
  if (!opts.sticky) setTimeout(kill, opts.ms || 3600);
  return kill;
}

/* ── layer navigation (Android Back closes the top layer) ───────────────── */

const LAYERS = ['player', 'chapters', 'reader', 'picker', 'help'];
const nav = {
  stack: [],
  has(n) { return this.stack.indexOf(n) >= 0; },
  open(n) {
    if (this.has(n)) return;
    this.stack = this.stack.concat(n);
    try { history.pushState({ pa: this.stack }, ''); } catch (e) {}
    apply();
  },
  close(n) {
    const i = this.stack.indexOf(n);
    if (i < 0) return;
    const steps = this.stack.length - i;
    try { history.go(-steps); } catch (e) { this.stack = this.stack.slice(0, i); apply(); }
  },
  closeTop() { if (this.stack.length) this.close(this.stack[this.stack.length - 1]); },
};

function apply() {
  const s = nav.stack;
  const on = (node, name) => {
    if (!node) return;
    const open = s.indexOf(name) >= 0;
    node.classList.toggle('is-open', open);
    node.setAttribute('aria-hidden', String(!open));
  };
  on(el.sheet, 'player');
  on(el.chapPanel, 'chapters');
  on(el.reader, 'reader');
  on(el.picker, 'picker');
  on(el.help, 'help');
  document.body.classList.toggle('no-scroll', s.length > 0);
}

window.addEventListener('popstate', e => {
  const st = e.state && Array.isArray(e.state.pa) ? e.state.pa : [];
  nav.stack = st.filter(n => LAYERS.indexOf(n) >= 0);
  apply();
});

/* ── library ────────────────────────────────────────────────────────────── */

function normalise(raw) {
  if (!raw || !Array.isArray(raw.lectures)) throw new Error('bad manifest');
  const base = String(raw.base || '').replace(/\/?$/, '/');
  const lectures = raw.lectures
    .filter(l => l && l.id && l.audio)
    .map((l, i) => ({
      id: String(l.id),
      n: Number(l.n) || i + 1,
      title: String(l.title || l.id),
      series: String(l.series || 'Lectures'),
      audio: String(l.audio),
      bytes: Number(l.bytes) || 0,
      duration: Number(l.duration) || 0,
      doc: l.doc ? String(l.doc) : null,
      bg: l.bg ? String(l.bg) : '',
      published: l.published || '',
      chapters: (Array.isArray(l.chapters) ? l.chapters : [])
        .filter(c => c && isFinite(Number(c.start)))
        .map(c => ({ title: String(c.title || ''), start: Math.max(0, Number(c.start)) }))
        .sort((a, b) => a.start - b.start),
    }));
  if (!lectures.length) throw new Error('empty manifest');
  return { title: raw.title || 'Papers, as Audio', base, revision: raw.revision || '', generated: raw.generated || '', lectures };
}

async function fetchLibrary() {
  try {
    const res = await fetch(LIBRARY_URL, { mode: 'cors', cache: 'no-cache' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const lib = normalise(await res.json());
    store.setJSON('library', lib);
    return { lib, source: 'network' };
  } catch (netErr) {
    const cached = store.getJSON('library', null);
    if (cached) {
      try { return { lib: normalise(cached), source: 'cache', err: netErr }; } catch (e) {}
    }
    try {
      const res = await fetch('./library.json', { cache: 'no-cache' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return { lib: normalise(await res.json()), source: 'bundled', err: netErr };
    } catch (e) {
      return { lib: null, source: 'none', err: netErr };
    }
  }
}

function markNew(ids) {
  const seen = store.getJSON('seen', null);
  if (!Array.isArray(seen)) { store.setJSON('seen', ids); return new Set(); }
  const known = new Set(seen);
  const fresh = new Set(ids.filter(id => !known.has(id)));
  ids.forEach(id => known.add(id));
  store.setJSON('seen', Array.from(known));
  return fresh;
}

function adoptLibrary(lib, source) {
  state.lib = lib;
  state.base = lib.base;
  state.lectures = lib.lectures;
  state.source = source;
  state.newIds = markNew(lib.lectures.map(l => l.id));
  el.offlinePill.hidden = source === 'network';
  el.offlinePill.textContent = source === 'bundled' ? 'bundled list' : 'offline';
  el.foot.textContent = lib.lectures.length
    ? `${lib.lectures.length} lectures · app ${APP_VERSION}${lib.revision ? ' · ' + String(lib.revision).slice(0, 7) : ''}`
    : '';
  renderList();
  renderContinue();
  refreshStorage();
}

/* ── library rendering ──────────────────────────────────────────────────── */

function skeletons(n) {
  let h = '';
  for (let i = 0; i < n; i++) {
    h += `<div class="sk"><div class="sk-b sk-art"></div><div class="sk-main">
      <div class="sk-b sk-l1" style="width:${60 + (i * 13) % 32}%"></div>
      <div class="sk-b sk-l2"></div></div></div>`;
  }
  return h;
}

function matches(l, q) {
  if (!q) return true;
  return l.title.toLowerCase().indexOf(q) >= 0
      || l.series.toLowerCase().indexOf(q) >= 0
      || String(l.n) === q
      || l.id.toLowerCase().indexOf(q) >= 0;
}

function rowHtml(l) {
  const p = progressOf(l);
  const done = isDone(l.id);
  const dl = state.downloaded.has(audioUrl(l));
  const chaps = l.chapters.length;
  const cur = state.i >= 0 && state.lectures[state.i].id === l.id;

  let statusBit = '';
  if (done) statusBit = `<i class="dot"></i><span class="done">✓ Finished</span>`;
  else if (p > 0.01) statusBit = `<i class="dot"></i><span>${esc(mins((l.duration || 0) * (1 - p)))} left</span>`;

  return `<button class="row${cur ? ' is-current' : ''}" type="button" role="listitem" data-id="${esc(l.id)}">
    <span class="row-art">
      <img src="${esc(artUrl(l.bg, 160))}" alt="" loading="lazy" decoding="async" width="50" height="50">
      <span class="n">${esc(String(l.n))}</span>
    </span>
    <span class="row-main">
      <span class="row-title">${esc(l.title)}</span>
      <span class="row-sub">
        <span>${esc(mins(l.duration))}</span>
        ${chaps ? `<i class="dot"></i><span>${chaps} chapters</span>` : ''}
        ${statusBit}
      </span>
      ${p > 0.005 && p < 0.995 ? `<span class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></span>` : ''}
    </span>
    ${state.newIds.has(l.id) ? '<span class="badge-new">NEW</span>' : ''}
    <span class="row-end${dl ? ' ok' : ''}" data-end>${dl ? '<svg class="ic"><use href="#i-download"></use></svg>' : ''}</span>
  </button>`;
}

function renderList() {
  if (!state.lectures.length) return;
  const q = el.search.value.trim().toLowerCase();
  const hits = state.lectures.filter(l => matches(l, q));

  if (!hits.length) {
    el.list.innerHTML = `<div class="empty"><b>Nothing matches “${esc(el.search.value.trim())}”</b>
      Try a different word, or the lecture number.</div>`;
    return;
  }

  let html = '', series = null;
  hits.forEach(l => {
    if (l.series !== series) {
      series = l.series;
      const count = hits.filter(x => x.series === series).length;
      html += `<h2 class="group-head">${esc(series)} <b>${count}</b></h2>`;
    }
    html += rowHtml(l);
  });
  el.list.innerHTML = html;
}

function renderContinue() {
  const lastId = store.get('last', '');
  let l = state.lectures.find(x => x.id === lastId);
  const resuming = !!l && savedPos(l.id) > 10 && !isDone(l.id);
  if (!l) l = state.lectures.find(x => !isDone(x.id)) || state.lectures[0];
  if (!l) { el.continueSlot.innerHTML = ''; return; }

  const p = progressOf(l);
  const left = (l.duration || 0) * (1 - p);
  const sub = resuming ? `${mins(left)} left · ${Math.round(p * 100)}% done`
            : isDone(l.id) ? 'Finished · play again'
            : `${mins(l.duration)} · ${l.series}`;

  el.continueSlot.innerHTML = `<button class="continue" type="button" data-id="${esc(l.id)}">
    <img class="continue-art" src="${esc(artUrl(l.bg, 200))}" alt="" decoding="async">
    <span class="continue-main">
      <span class="continue-eyebrow">${resuming ? 'Continue listening' : 'Start here'}</span>
      <span class="continue-title">${esc(l.title)}</span>
      <span class="continue-sub">${esc(sub)}</span>
    </span>
    <span class="continue-play" aria-hidden="true"><svg class="ic"><use href="#i-play"></use></svg></span>
  </button>`;
}

/* ── storage / downloads summary ────────────────────────────────────────── */

let estimateCache = null;
async function refreshStorage() {
  if (state.dl.size) {
    let loaded = 0, total = 0;
    state.dl.forEach(r => { loaded += r.loaded; total += r.total || 0; });
    const pct = total ? Math.round((loaded / total) * 100) : 0;
    el.storageLine.textContent = `Downloading ${state.dl.size} lecture${state.dl.size > 1 ? 's' : ''} · ${pct}%`;
    return;
  }
  const n = state.lectures.filter(l => state.downloaded.has(audioUrl(l))).length;
  let used = '';
  if (navigator.storage && navigator.storage.estimate) {
    try {
      estimateCache = await navigator.storage.estimate();
      if (estimateCache && estimateCache.usage) used = ` · ${bytesLabel(estimateCache.usage)} used`;
    } catch (e) {}
  }
  el.storageLine.textContent = state.lectures.length
    ? (n ? `${n} of ${state.lectures.length} available offline${used}`
         : `${state.lectures.length} lectures · nothing downloaded yet`)
    : 'Loading library…';

  const remaining = state.lectures.filter(l => !state.downloaded.has(audioUrl(l))).length;
  el.btnDownloadAll.textContent = state.bulk ? 'Stop' : 'Get all';
  el.btnDownloadAll.disabled = !state.bulk && remaining === 0;
}

async function scanDownloads() {
  if (!('caches' in window)) return;
  try {
    const cache = await caches.open(AUDIO_CACHE);
    const keys = await cache.keys();
    state.downloaded = new Set(keys.map(r => r.url));
  } catch (e) { /* cache unavailable */ }
}

function askPersist() {
  if (store.get('persistAsked', '') === '1') return;
  store.set('persistAsked', '1');
  if (navigator.storage && navigator.storage.persist) {
    navigator.storage.persist().catch(() => {});
  }
}

function updateRowDl(id) {
  const rec = state.dl.get(id);
  let row = null;
  const rows = el.list.querySelectorAll('.row');
  for (let i = 0; i < rows.length; i++) if (rows[i].getAttribute('data-id') === id) { row = rows[i]; break; }
  if (!row) return;
  const end = row.querySelector('[data-end]');
  if (!end) return;
  if (rec) {
    const pct = rec.total ? Math.round((rec.loaded / rec.total) * 100) : 0;
    end.classList.remove('ok');
    end.innerHTML = `<span style="font-size:10.5px;font-weight:700;color:var(--accent)">${pct}%</span>`;
  } else {
    const l = state.lectures.find(x => x.id === id);
    const dl = l && state.downloaded.has(audioUrl(l));
    end.classList.toggle('ok', !!dl);
    end.innerHTML = dl ? '<svg class="ic"><use href="#i-download"></use></svg>' : '';
  }
}

async function startDownload(l) {
  if (!l || state.dl.has(l.id)) return false;
  if (!('caches' in window)) { toast('Offline downloads need a secure (https) connection'); return false; }
  const url = audioUrl(l);
  if (state.downloaded.has(url)) return true;

  const ctrl = new AbortController();
  const rec = { loaded: 0, total: l.bytes || 0, ctrl };
  state.dl.set(l.id, rec);
  paintDownloadTool();
  updateRowDl(l.id);
  refreshStorage();

  let ok = false;
  try {
    const res = await fetch(url, { mode: 'cors', signal: ctrl.signal });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    rec.total = Number(res.headers.get('content-length')) || l.bytes || 0;

    const chunks = [];
    if (res.body && res.body.getReader) {
      const reader = res.body.getReader();
      let tick = 0;
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        chunks.push(step.value);
        rec.loaded += step.value.length;
        if (++tick % 4 === 0) { paintDownloadTool(); updateRowDl(l.id); refreshStorage(); }
      }
    } else {
      const buf = new Uint8Array(await res.arrayBuffer());
      chunks.push(buf); rec.loaded = buf.length;
    }

    const blob = new Blob(chunks, { type: 'audio/mpeg' });
    const cache = await caches.open(AUDIO_CACHE);
    await cache.put(url, new Response(blob, {
      status: 200,
      headers: {
        'Content-Type': 'audio/mpeg',
        'Content-Length': String(blob.size),
        'Accept-Ranges': 'bytes',
      },
    }));
    state.downloaded.add(url);
    askPersist();
    ok = true;
  } catch (err) {
    const name = String((err && err.name) || '');
    const msg = String((err && err.message) || err);
    if (name === 'AbortError') { /* silent — the user asked */ }
    else if (/quota/i.test(name + msg)) toast('Storage is full — remove a download and try again');
    else toast(`Could not download “${l.title}” (${msg})`);
  } finally {
    state.dl.delete(l.id);
    paintDownloadTool();
    updateRowDl(l.id);
    refreshStorage();
  }
  return ok;
}

async function removeDownload(l) {
  if (!('caches' in window)) return;
  try {
    const cache = await caches.open(AUDIO_CACHE);
    await cache.delete(audioUrl(l));
  } catch (e) {}
  state.downloaded.delete(audioUrl(l));
  paintDownloadTool();
  updateRowDl(l.id);
  refreshStorage();
  toast(`Removed the download for “${l.title}”`);
}

async function downloadAll() {
  if (state.bulk) {                       // second tap = stop
    state.bulk = false;
    state.dl.forEach(r => { try { r.ctrl.abort(); } catch (e) {} });
    refreshStorage();
    return;
  }
  const queue = state.lectures.filter(l => !state.downloaded.has(audioUrl(l)));
  if (!queue.length) return;
  state.bulk = true;
  refreshStorage();
  let n = 0;
  for (const l of queue) {
    if (!state.bulk) break;
    if (await startDownload(l)) n++;
  }
  state.bulk = false;
  renderList();
  refreshStorage();
  toast(n ? `Downloaded ${n} lecture${n > 1 ? 's' : ''}` : 'Nothing was downloaded');
}

/* ── playback ───────────────────────────────────────────────────────────── */

function durationOf() {
  if (isFinite(audio.duration) && audio.duration > 0) return audio.duration;
  const l = current();
  return l && l.duration ? l.duration : 0;
}

function loadLecture(i, autoplay, seekTo) {
  if (!state.lectures.length) return;
  const n = state.lectures.length;
  i = ((i % n) + n) % n;
  if (state.i >= 0 && state.i !== i) savePosition();

  state.i = i;
  const l = state.lectures[i];
  state.chapI = -1;
  state.pendingSeek = (seekTo != null) ? seekTo : savedPos(l.id);

  audio.pause();
  audio.src = audioUrl(l);
  audio.playbackRate = state.rate;
  try { audio.load(); } catch (e) {}

  store.set('last', l.id);
  paintNowPlaying();
  renderList();
  renderContinue();

  if (autoplay) {
    const p = audio.play();
    if (p && p.catch) p.catch(() => {});
  }
}

function play(id, opts) {
  opts = opts || {};
  const i = state.lectures.findIndex(l => l.id === id);
  if (i < 0) return;
  if (state.i === i && audio.src) {
    if (audio.paused) { const p = audio.play(); if (p && p.catch) p.catch(() => {}); }
  } else {
    loadLecture(i, true);
  }
  if (opts.openSheet !== false) nav.open('player');
}

function togglePlay() {
  if (!current()) return;
  if (audio.paused) { const p = audio.play(); if (p && p.catch) p.catch(() => {}); }
  else audio.pause();
}
function seekTo(t) {
  const d = durationOf();
  if (!d) return;
  try { audio.currentTime = clamp(t, 0, Math.max(0, d - 0.25)); } catch (e) {}
  paintTime();
}
const skip = delta => seekTo((audio.currentTime || 0) + delta);

function savePosition() {
  const l = current();
  if (!l) return;
  const t = audio.currentTime, d = durationOf();
  if (isFinite(t) && d) {
    if (t > 5 && t < d - 8) store.set(posKey(l.id), t.toFixed(1));
    else if (t <= 5) store.del(posKey(l.id));
  }
  store.set('last', l.id);
}

/* ── chapters ───────────────────────────────────────────────────────────── */

const chaptersOf = l => (l && l.chapters) || [];

function chapterIndexAt(chs, t) {
  if (!chs.length) return -1;
  let i = 0;
  for (let k = 0; k < chs.length; k++) {
    if (t + 0.25 >= chs[k].start) i = k; else break;
  }
  return i;
}

function renderTicks() {
  const chs = chaptersOf(current());
  const d = durationOf();
  if (!chs.length || !d) { el.ticks.innerHTML = ''; return; }
  el.ticks.innerHTML = chs
    .filter(c => c.start > 0.5 && c.start < d - 0.5)
    .map(c => `<i style="left:${((c.start / d) * 100).toFixed(3)}%"></i>`)
    .join('');
}

function renderChapterList() {
  const l = current();
  const chs = chaptersOf(l);
  if (!chs.length) { el.chapItems.innerHTML = ''; return; }
  el.chapItems.innerHTML = chs.map((c, k) => `
    <button class="chap" type="button" data-start="${c.start}" data-k="${k}">
      <span class="chap-n">${k + 1}</span>
      <span class="chap-t">${esc(c.title || 'Chapter ' + (k + 1))}</span>
      <span class="chap-s">${esc(hms(c.start))}</span>
    </button>`).join('');
  paintChapterActive(true);
}

function paintChapterActive(scrollIntoView) {
  const items = el.chapItems.querySelectorAll('.chap');
  if (!items.length) return;
  let active = null;
  for (let k = 0; k < items.length; k++) {
    const on = k === state.chapI;
    items[k].classList.toggle('is-on', on);
    items[k].classList.toggle('is-past', k < state.chapI);
    if (on) active = items[k];
  }
  if (scrollIntoView && active) {
    const box = el.chapItems;
    box.scrollTop = Math.max(0, active.offsetTop - box.clientHeight / 2 + active.offsetHeight / 2);
  }
}

function updateChapter(force) {
  const chs = chaptersOf(current());
  if (!chs.length) return;
  const i = chapterIndexAt(chs, audio.currentTime || 0);
  if (i === state.chapI && !force) return;
  state.chapI = i;
  el.chapNow.textContent = (chs[i] && chs[i].title) || `Chapter ${i + 1}`;
  el.btnChapPrev.disabled = false;
  el.btnChapNext.disabled = i >= chs.length - 1;
  paintChapterActive(false);
}

function gotoChapter(dir) {
  const chs = chaptersOf(current());
  if (!chs.length) return;
  const i = chapterIndexAt(chs, audio.currentTime || 0);
  if (dir < 0) {
    const atStart = (audio.currentTime || 0) - chs[i].start <= 3;
    seekTo(chs[atStart && i > 0 ? i - 1 : i].start);
  } else if (i + 1 < chs.length) {
    seekTo(chs[i + 1].start);
  }
  updateChapter(true);
}

/* ── now playing painting ───────────────────────────────────────────────── */

function paintNowPlaying() {
  const l = current();
  if (!l) return;

  const big = artUrl(l.bg, 800);
  el.cover.src = big;
  el.cover.alt = `Artwork for ${l.title}`;
  el.sheetBg.style.backgroundImage = `url("${big}")`;
  el.npSeries.textContent = l.series;
  el.npTitle.textContent = l.title;
  el.sheetNum.textContent = `${l.n} / ${state.lectures.length}`;
  el.rateVal.textContent = `${state.rate}×`;

  el.miniArt.src = artUrl(l.bg, 120);
  el.miniTitle.textContent = l.title;
  el.miniSub.textContent = l.series;
  el.mini.hidden = false;

  el.btnRead.disabled = !l.doc;
  el.btnPrev.disabled = state.lectures.length < 2;
  el.btnNext.disabled = state.lectures.length < 2;

  const chs = chaptersOf(l);
  el.chapBar.hidden = chs.length === 0;
  el.chapNow.textContent = chs.length ? (chs[0].title || 'Chapter 1') : 'Chapters';
  renderChapterList();
  renderTicks();

  paintDownloadTool();
  paintTime();
  updateMediaSession();
  if (chs.length) updateChapter(true);
}

function paintTime() {
  const d = durationOf();
  const t = clamp(audio.currentTime || 0, 0, d || 0);
  el.tCur.textContent = hms(t);
  el.tTotal.textContent = d ? hms(d) : '—';
  el.tRemain.textContent = d ? '-' + hms(Math.max(0, d - t)) : '-0:00';
  const p = d ? (t / d) * 100 : 0;
  if (!state.scrubbing) {
    el.scrub.value = String(Math.round(p * 10));
    el.scrub.style.setProperty('--p', p.toFixed(2));
  }
  el.miniFill.style.width = p.toFixed(2) + '%';
}

function paintPlayState() {
  const playing = !audio.paused && !audio.ended;
  setIcon(el.btnPlay, playing ? 'i-pause' : 'i-play');
  setIcon(el.miniPlay, playing ? 'i-pause' : 'i-play');
  el.btnPlay.setAttribute('aria-label', playing ? 'Pause' : 'Play');
  if ('mediaSession' in navigator) {
    try { navigator.mediaSession.playbackState = playing ? 'playing' : 'paused'; } catch (e) {}
  }
}

function paintDownloadTool() {
  const l = current();
  const btn = el.btnDownload;
  if (!l) return;
  const rec = state.dl.get(l.id);
  const have = state.downloaded.has(audioUrl(l));
  btn.classList.toggle('is-busy', !!rec);
  btn.classList.toggle('is-done', !rec && have);
  if (rec) {
    const p = rec.total ? rec.loaded / rec.total : 0;
    btn.querySelector('.tool-ic').style.setProperty('--dl', p.toFixed(4));
    setIcon(el.dlIcon, 'i-close');
    el.dlLabel.textContent = `${Math.round(p * 100)}%`;
  } else if (have) {
    setIcon(el.dlIcon, 'i-check');
    el.dlLabel.textContent = 'Saved';
  } else {
    setIcon(el.dlIcon, 'i-download');
    el.dlLabel.textContent = l.bytes ? bytesLabel(l.bytes) : 'Download';
  }
}

function updateMediaSession() {
  const l = current();
  if (!l || !('mediaSession' in navigator)) return;
  try {
    if (window.MediaMetadata) {
      const art = artUrl(l.bg, 512);
      navigator.mediaSession.metadata = new MediaMetadata({
        title: l.title,
        artist: 'Papers, as Audio',
        album: l.series,
        artwork: [
          { src: artUrl(l.bg, 256), sizes: '256x256', type: 'image/jpeg' },
          { src: art, sizes: '512x512', type: 'image/jpeg' },
        ],
      });
    }
  } catch (e) {}
}

let lastPosState = 0;
function updatePositionState() {
  if (!('mediaSession' in navigator) || !navigator.mediaSession.setPositionState) return;
  const now = Date.now();
  if (now - lastPosState < 900) return;
  lastPosState = now;
  const d = durationOf();
  if (!d) return;
  try {
    navigator.mediaSession.setPositionState({
      duration: d,
      position: clamp(audio.currentTime || 0, 0, d),
      playbackRate: audio.playbackRate || 1,
    });
  } catch (e) {}
}

if ('mediaSession' in navigator) {
  const ms = navigator.mediaSession;
  const set = (a, fn) => { try { ms.setActionHandler(a, fn); } catch (e) {} };
  set('play', () => togglePlay());
  set('pause', () => audio.pause());
  set('stop', () => { audio.pause(); savePosition(); });
  set('seekbackward', d => skip(-((d && d.seekOffset) || SKIP_BACK)));
  set('seekforward', d => skip((d && d.seekOffset) || SKIP_FWD));
  set('previoustrack', () => loadLecture(state.i - 1, true));
  set('nexttrack', () => loadLecture(state.i + 1, true));
  set('seekto', d => { if (d && d.seekTime != null) seekTo(d.seekTime); });
}

/* ── audio events ───────────────────────────────────────────────────────── */

audio.addEventListener('loadedmetadata', () => {
  const t = state.pendingSeek;
  state.pendingSeek = null;
  const d = durationOf();
  if (t != null && t > 1 && d && t < d - 8) { try { audio.currentTime = t; } catch (e) {} }
  audio.playbackRate = state.rate;
  renderTicks();
  paintTime();
  updateChapter(true);
});
audio.addEventListener('timeupdate', () => {
  paintTime();
  updatePositionState();
  updateChapter(false);
});
audio.addEventListener('durationchange', () => { renderTicks(); paintTime(); });
audio.addEventListener('play', paintPlayState);
audio.addEventListener('pause', () => { paintPlayState(); savePosition(); });
audio.addEventListener('ratechange', () => { el.rateVal.textContent = `${audio.playbackRate}×`; });
audio.addEventListener('ended', () => {
  const l = current();
  if (l) { store.del(posKey(l.id)); store.set(doneKey(l.id), '1'); }
  renderList(); renderContinue();
  if (state.sleep.mode === 'end') { setSleep('off'); toast('Sleep timer: stopped at the end of the lecture'); return; }
  if (state.i + 1 < state.lectures.length) loadLecture(state.i + 1, true);
  else { paintPlayState(); toast('That was the last lecture.'); }
});
audio.addEventListener('error', () => {
  const l = current();
  if (!l || !audio.src) return;
  const code = audio.error && audio.error.code;
  const offline = !navigator.onLine;
  const saved = state.downloaded.has(audioUrl(l));
  let msg = 'Could not play this lecture.';
  if (offline && !saved) msg = 'You are offline and this lecture is not downloaded.';
  else if (code === 2) msg = 'Network problem while streaming — check your connection.';
  else if (code === 4) msg = 'This audio file could not be loaded.';
  toast(msg, { ms: 5200 });
  paintPlayState();
});

setInterval(() => { if (!audio.paused) savePosition(); }, 5000);
window.addEventListener('pagehide', savePosition);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') savePosition();
  else onForeground();
});

/* ── sleep timer ────────────────────────────────────────────────────────── */

let sleepTick = null;
function setSleep(v) {
  if (sleepTick) { clearInterval(sleepTick); sleepTick = null; }
  if (v === 'off') {
    state.sleep = { mode: 'off', deadline: 0 };
    el.btnSleep.classList.remove('is-on');
    el.sleepLabel.textContent = 'Sleep';
    return;
  }
  el.btnSleep.classList.add('is-on');
  if (v === 'end') {
    state.sleep = { mode: 'end', deadline: 0 };
    el.sleepLabel.textContent = 'End';
    return;
  }
  const secs = Number(v) || 0;
  state.sleep = { mode: 'timer', deadline: Date.now() + secs * 1000, secs };
  const paint = () => {
    const left = Math.max(0, state.sleep.deadline - Date.now());
    el.sleepLabel.textContent = `${Math.ceil(left / 60000)} min`;
    if (left <= 0) {
      audio.pause();
      setSleep('off');
      toast('Sleep timer — playback paused');
    }
  };
  paint();
  sleepTick = setInterval(paint, 1000);
}

function openSleepPicker() {
  el.pickerTitle.textContent = 'Sleep timer';
  el.pickerOpts.innerHTML = SLEEPS.map(o => {
    const on = (o.v === 'off' && state.sleep.mode === 'off')
            || (o.v === 'end' && state.sleep.mode === 'end')
            || (state.sleep.mode === 'timer' && Number(o.v) === state.sleep.secs);
    return `<button class="opt${on ? ' is-on' : ''}" type="button" data-v="${o.v}">
      <span>${esc(o.label)}</span>${on ? '<svg class="ic"><use href="#i-check"></use></svg>' : ''}</button>`;
  }).join('');
  nav.open('picker');
}

/* ── transcript reader ──────────────────────────────────────────────────── */

const docMem = new Map();

async function fetchDoc(url) {
  if (docMem.has(url)) return docMem.get(url);
  let cache = null;
  if ('caches' in window) { try { cache = await caches.open(DOC_CACHE); } catch (e) {} }

  if (cache) {
    const hit = await cache.match(url).catch(() => null);
    if (hit) {
      const text = await hit.text();
      docMem.set(url, text);
      fetch(url, { mode: 'cors' })
        .then(r => { if (r.ok) cache.put(url, r.clone()).catch(() => {}); })
        .catch(() => {});
      return text;
    }
  }
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  const text = await res.text();
  docMem.set(url, text);
  if (cache) {
    cache.put(url, new Response(text, { headers: { 'Content-Type': 'text/markdown; charset=utf-8' } })).catch(() => {});
  }
  return text;
}

async function renderMarkdown(text) {
  try {
    await loadScript(MARKED_URL);
  } catch (e) { /* fall through to the plain-text renderer */ }
  if (window.marked) {
    const fn = typeof window.marked.parse === 'function' ? window.marked.parse : window.marked;
    try { return fn(text, { breaks: false, gfm: true }); } catch (e) {}
  }
  return `<pre>${esc(text)}</pre>`;
}

async function typeset(node) {
  try {
    await loadCss(KATEX_CSS);
    await loadScript(KATEX_JS);
    await loadScript(KATEX_AUTO);
    if (window.renderMathInElement) {
      window.renderMathInElement(node, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      });
    }
  } catch (e) { /* maths just stays as source text */ }
}

async function openReader() {
  const l = current();
  if (!l || !l.doc) return;
  nav.open('reader');
  el.readerTitle.textContent = l.title;
  el.readerBody.innerHTML = '<p class="reader-note">Loading transcript…</p>';
  try {
    const text = await fetchDoc(fullUrl(l.doc));
    el.readerBody.innerHTML = await renderMarkdown(text);
    el.readerBody.scrollTop = 0;
    if (/\\\(|\\\[|\$\$/.test(text)) await typeset(el.readerBody);
  } catch (err) {
    el.readerBody.innerHTML = `<p class="reader-note">The transcript could not be loaded${
      navigator.onLine ? '' : ' — you are offline'}.<br>Open it once while online and it is kept for later.</p>`;
  }
}

function setReaderFont(px) {
  const v = clamp(px, 14, 26);
  el.readerBody.style.setProperty('--reader-font', v + 'px');
  store.set('readerFont', v);
  return v;
}

/* ── theme ──────────────────────────────────────────────────────────────── */

function effectiveTheme() {
  const pinned = document.documentElement.getAttribute('data-theme');
  if (pinned === 'light' || pinned === 'dark') return pinned;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
function paintTheme() {
  const t = effectiveTheme();
  setIcon(el.btnTheme, t === 'dark' ? 'i-moon' : 'i-sun');
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', t === 'dark' ? '#0d1017' : '#f4f6fa');
}
function toggleTheme() {
  const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  store.set('theme', next);
  paintTheme();
}

/* ── service worker + install ───────────────────────────────────────────── */

function showUpdateToast(worker) {
  toast('A new version is ready.', {
    sticky: true,
    action: 'Reload',
    onAction: () => {
      state.updateAccepted = true;
      try { worker.postMessage({ type: 'SKIP_WAITING' }); } catch (e) { location.reload(); }
      setTimeout(() => { if (!document.hidden) location.reload(); }, 1800);
    },
  });
}

function registerSW() {
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!state.updateAccepted) return;
    state.updateAccepted = false;
    location.reload();
  });
  navigator.serviceWorker.register('./sw.js', { scope: './' }).then(reg => {
    state.swReg = reg;
    if (reg.waiting && navigator.serviceWorker.controller) showUpdateToast(reg.waiting);
    reg.addEventListener('updatefound', () => {
      const nw = reg.installing;
      if (!nw) return;
      nw.addEventListener('statechange', () => {
        if (nw.state === 'installed' && navigator.serviceWorker.controller) showUpdateToast(nw);
      });
    });
  }).catch(() => {});
}

let libRefreshedAt = 0;
async function onForeground() {
  if (state.swReg) { try { state.swReg.update(); } catch (e) {} }
  if (Date.now() - libRefreshedAt < 60000) return;
  libRefreshedAt = Date.now();
  const before = state.lectures.length;
  const { lib, source } = await fetchLibrary();
  if (lib && source === 'network') {
    adoptLibrary(lib, source);
    if (lib.lectures.length > before) toast(`${lib.lectures.length - before} new lecture(s) added`);
  }
}

function isStandalone() {
  return (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches)
      || window.navigator.standalone === true;
}

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  state.installPrompt = e;
  if (!isStandalone()) el.btnInstall.hidden = false;
});
window.addEventListener('appinstalled', () => {
  state.installPrompt = null;
  el.btnInstall.hidden = true;
  toast('Installed — open it from your home screen');
});

/* ── events ─────────────────────────────────────────────────────────────── */

el.list.addEventListener('click', e => {
  const row = e.target.closest ? e.target.closest('.row') : null;
  if (!row) return;
  play(row.getAttribute('data-id'));
});
el.continueSlot.addEventListener('click', e => {
  const c = e.target.closest ? e.target.closest('.continue') : null;
  if (!c) return;
  play(c.getAttribute('data-id'));
});

el.search.addEventListener('input', () => {
  el.searchClear.hidden = !el.search.value;
  renderList();
});
el.searchClear.addEventListener('click', () => {
  el.search.value = ''; el.searchClear.hidden = true; renderList(); el.search.focus();
});
el.btnDownloadAll.addEventListener('click', downloadAll);

el.miniOpen.addEventListener('click', () => nav.open('player'));
el.miniPlay.addEventListener('click', togglePlay);
el.miniFwd.addEventListener('click', () => skip(SKIP_FWD));

el.btnCloseSheet.addEventListener('click', () => nav.close('player'));
el.btnPlay.addEventListener('click', togglePlay);
el.btnBack.addEventListener('click', () => skip(-SKIP_BACK));
el.btnFwd.addEventListener('click', () => skip(SKIP_FWD));
el.btnPrev.addEventListener('click', () => loadLecture(state.i - 1, true));
el.btnNext.addEventListener('click', () => loadLecture(state.i + 1, true));

el.btnRate.addEventListener('click', () => {
  const i = RATES.indexOf(state.rate);
  state.rate = RATES[(i + 1) % RATES.length];
  audio.playbackRate = state.rate;
  el.rateVal.textContent = `${state.rate}×`;
  store.set('rate', state.rate);
});
el.btnSleep.addEventListener('click', openSleepPicker);
el.btnRead.addEventListener('click', openReader);
el.btnDownload.addEventListener('click', () => {
  const l = current();
  if (!l) return;
  const rec = state.dl.get(l.id);
  if (rec) { try { rec.ctrl.abort(); } catch (e) {} return; }
  if (state.downloaded.has(audioUrl(l))) removeDownload(l);
  else startDownload(l).then(ok => { if (ok) { renderList(); toast(`“${l.title}” is available offline`); } });
});

el.btnChapters.addEventListener('click', () => { renderChapterList(); nav.open('chapters'); });
el.chapClose.addEventListener('click', () => nav.close('chapters'));
el.btnChapPrev.addEventListener('click', () => gotoChapter(-1));
el.btnChapNext.addEventListener('click', () => gotoChapter(1));
el.chapItems.addEventListener('click', e => {
  const b = e.target.closest ? e.target.closest('.chap') : null;
  if (!b) return;
  seekTo(parseFloat(b.getAttribute('data-start')) || 0);
  updateChapter(true);
  nav.close('chapters');
});

/* scrubber */
function scrubValueToTime() {
  const d = durationOf();
  return d ? (Number(el.scrub.value) / 1000) * d : 0;
}
['pointerdown', 'touchstart', 'keydown'].forEach(ev =>
  el.scrub.addEventListener(ev, () => { state.scrubbing = true; el.scrub.classList.add('is-scrubbing'); }, { passive: true }));
el.scrub.addEventListener('input', () => {
  state.scrubbing = true;
  const t = scrubValueToTime(), d = durationOf();
  el.scrub.style.setProperty('--p', (Number(el.scrub.value) / 10).toFixed(2));
  el.tCur.textContent = hms(t);
  el.tRemain.textContent = '-' + hms(Math.max(0, d - t));
});
function endScrub() {
  if (!state.scrubbing) return;
  state.scrubbing = false;
  el.scrub.classList.remove('is-scrubbing');
  seekTo(scrubValueToTime());
  updateChapter(true);
}
el.scrub.addEventListener('change', endScrub);
['pointerup', 'pointercancel', 'touchend', 'touchcancel'].forEach(ev =>
  el.scrub.addEventListener(ev, endScrub));

/* swipe the sheet down to dismiss */
(() => {
  let startY = null, dy = 0;
  const targets = [el.sheetGrab, el.coverWrap];
  const down = e => {
    if (nav.has('chapters')) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    startY = e.clientY; dy = 0;
    el.sheet.classList.add('is-dragging');
  };
  const move = e => {
    if (startY === null) return;
    dy = Math.max(0, e.clientY - startY);
    if (dy > 0) el.sheet.style.transform = `translateY(${dy}px)`;
  };
  const up = () => {
    if (startY === null) return;
    const d = dy; startY = null; dy = 0;
    el.sheet.classList.remove('is-dragging');
    el.sheet.style.transform = '';
    if (d > 110) nav.close('player');
  };
  targets.forEach(t => t && t.addEventListener('pointerdown', down));
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', up);
})();

/* picker / help / reader chrome */
el.pickerOpts.addEventListener('click', e => {
  const b = e.target.closest ? e.target.closest('.opt') : null;
  if (!b) return;
  setSleep(b.getAttribute('data-v'));
  nav.close('picker');
});
el.picker.addEventListener('click', e => { if (e.target === el.picker) nav.close('picker'); });
el.help.addEventListener('click', e => { if (e.target === el.help) nav.close('help'); });
el.helpClose.addEventListener('click', () => nav.close('help'));
el.btnHelp.addEventListener('click', () => nav.open('help'));
el.btnTheme.addEventListener('click', toggleTheme);
el.btnInstall.addEventListener('click', async () => {
  const p = state.installPrompt;
  if (!p) { toast('Use your browser menu → “Add to Home screen”'); return; }
  state.installPrompt = null;
  el.btnInstall.hidden = true;
  try { p.prompt(); await p.userChoice; } catch (e) {}
});

el.readerClose.addEventListener('click', () => nav.close('reader'));
let readerFont = parseFloat(store.get('readerFont', '17')) || 17;
el.fontUp.addEventListener('click', () => { readerFont = setReaderFont(readerFont + 1); });
el.fontDn.addEventListener('click', () => { readerFont = setReaderFont(readerFont - 1); });

/* keyboard */
document.addEventListener('keydown', e => {
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(tag);

  if (e.key === 'Escape') {
    if (nav.stack.length) { e.preventDefault(); nav.closeTop(); }
    else if (typing) document.activeElement.blur();
    return;
  }
  if (typing) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  switch (e.key) {
    case ' ': case 'k': e.preventDefault(); togglePlay(); break;
    case 'ArrowRight': e.preventDefault(); e.shiftKey ? loadLecture(state.i + 1, true) : skip(SKIP_FWD); break;
    case 'ArrowLeft':  e.preventDefault(); e.shiftKey ? loadLecture(state.i - 1, true) : skip(-SKIP_BACK); break;
    case ']': e.preventDefault(); el.btnRate.click(); break;
    case '[': {
      e.preventDefault();
      const i = RATES.indexOf(state.rate);
      state.rate = RATES[(i - 1 + RATES.length) % RATES.length];
      audio.playbackRate = state.rate;
      el.rateVal.textContent = `${state.rate}×`;
      store.set('rate', state.rate);
      break;
    }
    case 'r': case 'R': nav.has('reader') ? nav.close('reader') : openReader(); break;
    case 't': case 'T': toggleTheme(); break;
    case 'n': case 'N': loadLecture(state.i + 1, true); break;
    case 'p': case 'P': loadLecture(state.i - 1, true); break;
    case '/': e.preventDefault(); nav.stack.length && nav.closeTop(); el.search.focus(); break;
    case '?': e.preventDefault(); nav.has('help') ? nav.close('help') : nav.open('help'); break;
    default:
      if (/^[0-9]$/.test(e.key)) {
        const d = durationOf();
        if (d) seekTo((parseInt(e.key, 10) / 10) * d);
      }
  }
});

/* artwork URLs can rot — fall back rather than showing a broken-image glyph */
document.addEventListener('error', e => {
  const t = e.target;
  if (!t || t.tagName !== 'IMG') return;
  if (t === el.cover) {
    if (t.src.indexOf('icon-512') < 0) t.src = './icons/icon-512.png';
    return;
  }
  t.classList.add('is-broken');
}, true);

window.addEventListener('online', () => { el.offlinePill.hidden = state.source === 'network'; });
window.addEventListener('offline', () => { el.offlinePill.hidden = false; el.offlinePill.textContent = 'offline'; });

/* sticky group headings need the live top-bar height */
function measureTopbar() {
  const h = el.topbar.getBoundingClientRect().height;
  document.documentElement.style.setProperty('--top-h', Math.round(h) + 'px');
}
if (window.ResizeObserver) new ResizeObserver(measureTopbar).observe(el.topbar);
window.addEventListener('resize', measureTopbar);

/* ── boot ───────────────────────────────────────────────────────────────── */

async function boot() {
  try { history.replaceState({ pa: [] }, ''); } catch (e) {}
  apply();
  measureTopbar();
  paintTheme();
  setReaderFont(readerFont);
  el.rateVal.textContent = `${state.rate}×`;
  audio.playbackRate = state.rate;
  el.list.innerHTML = skeletons(7);
  if (isStandalone()) el.btnInstall.hidden = true;

  registerSW();
  await scanDownloads();

  const { lib, source, err } = await fetchLibrary();
  if (!lib) {
    el.list.innerHTML = `<div class="empty"><b>The library could not be loaded</b>
      ${navigator.onLine ? 'The lecture list is temporarily unavailable.' : 'You appear to be offline.'}
      <button class="btn-wide" type="button" id="retry">Try again</button></div>`;
    const r = $('#retry');
    if (r) r.addEventListener('click', () => location.reload());
    el.storageLine.textContent = 'Library unavailable';
    return;
  }
  adoptLibrary(lib, source);
  if (source !== 'network') {
    toast(source === 'cache' ? 'Showing the last saved lecture list' : 'Showing the bundled lecture list');
  }

  /* restore the last lecture without autoplaying */
  const lastId = store.get('last', '');
  const i = state.lectures.findIndex(l => l.id === lastId);
  if (i >= 0) loadLecture(i, false);

  paintPlayState();
}

boot();

})();

/* ==========================================================================
   Papers, as Audio — service worker

   Three jobs:
     1. precache the app shell so the PWA opens offline  (shell-<APP_VERSION>)
     2. serve downloaded lectures out of `audio-v1`, WITH correct Range
        support — Chrome's media element cannot seek without 206 responses
     3. stay out of the way of everything else (Hugging Face manifest fetches
        are passed straight through)

   Keep APP_VERSION in step with config.js on every release.
   ========================================================================== */

const APP_VERSION  = '2026.09.16-1';

const SHELL_CACHE = `shell-${APP_VERSION}`;
const AUDIO_CACHE = 'audio-v1';
const DOC_CACHE   = 'docs-v1';
const ART_CACHE   = 'art-v1';
const ART_MAX     = 60;

const SHELL = [
  './',
  './index.html',
  './app.css',
  './app.js',
  './config.js',
  './library.json',
  './manifest.webmanifest',
  './icons/icon.svg',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/maskable-512.png',
  './icons/apple-touch-icon.png',
];

const INDEX_URL = new URL('./index.html', self.location).href;
const SHELL_URLS = new Set(SHELL.map(p => new URL(p, self.location).href));

/* ──────────────────────────────────────────────────────────────────────────
   Pure Range helper — kept standalone so it can be unit-tested in isolation.
   Returns { start, end } (inclusive, clamped) or null when the header is
   absent / malformed / unsatisfiable (caller answers 416).
   ────────────────────────────────────────────────────────────────────────── */
function parseRange(header, total) {
  if (typeof header !== 'string' || !(total > 0)) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m) return null;

  const hasStart = m[1] !== '';
  const hasEnd = m[2] !== '';
  if (!hasStart && !hasEnd) return null;

  let start, end;
  if (!hasStart) {
    // suffix form: "bytes=-500" == the last 500 bytes
    const n = parseInt(m[2], 10);
    if (!(n > 0)) return null;
    start = Math.max(0, total - n);
    end = total - 1;
  } else {
    start = parseInt(m[1], 10);
    if (!Number.isFinite(start) || start >= total) return null;   // unsatisfiable
    end = hasEnd ? parseInt(m[2], 10) : total - 1;
    if (!Number.isFinite(end)) return null;
    if (end >= total) end = total - 1;
    if (end < start) return null;
  }
  return { start, end };
}

/* ── install ────────────────────────────────────────────────────────────── */

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    try {
      await cache.addAll(SHELL);
    } catch (e) {
      // one bad entry must not sink the whole install
      await Promise.all(SHELL.map(p => cache.add(p).catch(() => {})));
    }
  })());
});

/* ── activate ───────────────────────────────────────────────────────────── */

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const keep = new Set([SHELL_CACHE, AUDIO_CACHE, DOC_CACHE, ART_CACHE]);
    const names = await caches.keys();
    await Promise.all(names.map(n => (keep.has(n) ? null : caches.delete(n))));
    if (self.registration.navigationPreload) {
      try { await self.registration.navigationPreload.disable(); } catch (e) {}
    }
    await self.clients.claim();
  })());
});

self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting();
});

/* ── responses ──────────────────────────────────────────────────────────── */

function rangeResponse(blob, range, type) {
  const total = blob.size;
  const body = blob.slice(range.start, range.end + 1);
  return new Response(body, {
    status: 206,
    statusText: 'Partial Content',
    headers: {
      'Content-Type': type || 'audio/mpeg',
      'Content-Length': String(range.end - range.start + 1),
      'Content-Range': `bytes ${range.start}-${range.end}/${total}`,
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-store',
    },
  });
}

async function serveCachedMedia(request, cached) {
  const type = cached.headers.get('Content-Type') || 'audio/mpeg';
  const header = request.headers.get('range');
  const blob = await cached.blob();

  if (!header) {
    return new Response(blob, {
      status: 200,
      headers: {
        'Content-Type': type,
        'Content-Length': String(blob.size),
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
      },
    });
  }

  const range = parseRange(header, blob.size);
  if (!range) {
    return new Response(null, {
      status: 416,
      statusText: 'Range Not Satisfiable',
      headers: { 'Content-Range': `bytes */${blob.size}`, 'Accept-Ranges': 'bytes' },
    });
  }
  return rangeResponse(blob, range, type);
}

/* ── routing ────────────────────────────────────────────────────────────── */

const isMedia = (request, url) =>
  request.destination === 'audio' || /\.(mp3|m4a|ogg|opus|wav)$/i.test(url.pathname);

const isArtwork = url =>
  url.hostname === 'images.unsplash.com' || url.hostname === 'plus.unsplash.com';

async function shellFirst(request, url) {
  const cache = await caches.open(SHELL_CACHE);
  const hit = await cache.match(url.href, { ignoreVary: true });
  if (hit) {
    // keep the bundled fallback manifest fresh in the background
    if (url.href.endsWith('library.json')) {
      fetch(request).then(res => { if (res && res.ok) cache.put(url.href, res.clone()); }).catch(() => {});
    }
    return hit;
  }
  try {
    const res = await fetch(request);
    if (res && res.ok && res.type === 'basic') cache.put(url.href, res.clone()).catch(() => {});
    return res;
  } catch (e) {
    const idx = await cache.match(INDEX_URL, { ignoreVary: true });
    if (idx && request.destination === 'document') return idx;
    throw e;
  }
}

async function navigate(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const res = await fetch(request);
    if (res && res.ok) return res;
    throw new Error('bad status');
  } catch (e) {
    const hit = await cache.match(INDEX_URL, { ignoreVary: true })
             || await cache.match(new URL('./', self.location).href, { ignoreVary: true });
    if (hit) return hit;
    return new Response('<h1>Offline</h1>', { status: 503, headers: { 'Content-Type': 'text/html' } });
  }
}

async function trimCache(cache, max) {
  const keys = await cache.keys();
  for (let i = 0; i < keys.length - max; i++) await cache.delete(keys[i]);
}

async function artwork(request, url) {
  const cache = await caches.open(ART_CACHE);
  const hit = await cache.match(request, { ignoreVary: true });
  if (hit) return hit;
  try {
    const res = await fetch(request);
    const w = Number(url.searchParams.get('w') || 0);
    if (res && (res.ok || res.type === 'opaque') && w > 0 && w <= 800) {
      cache.put(request, res.clone()).then(() => trimCache(cache, ART_MAX)).catch(() => {});
    }
    return res;
  } catch (e) {
    const any = await cache.match(request, { ignoreVary: true });
    if (any) return any;
    throw e;
  }
}

/** Downloaded audio and previously-read transcripts come out of their caches;
 *  everything else (library.json above all) is passed straight through. */
async function cachedOrNetwork(request, url) {
  const audioCache = await caches.open(AUDIO_CACHE);
  // match on the URL string, never on the Request: a Request carrying a Range
  // header makes cache.match() synthesise its own partial response.
  const cachedAudio = await audioCache.match(url.href, { ignoreVary: true });
  if (cachedAudio) return serveCachedMedia(request, cachedAudio);

  if (isMedia(request, url)) return fetch(request);   // streaming — hands off

  try {
    return await fetch(request);
  } catch (err) {
    const docCache = await caches.open(DOC_CACHE);
    const hit = await docCache.match(url.href, { ignoreVary: true });
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;

  let url;
  try { url = new URL(request.url); } catch (e) { return; }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  if (request.mode === 'navigate') {
    event.respondWith(navigate(request));
    return;
  }

  if (isArtwork(url)) { event.respondWith(artwork(request, url)); return; }

  if (url.origin === self.location.origin) {
    if (SHELL_URLS.has(url.href)) { event.respondWith(shellFirst(request, url)); return; }
    // audio is normally cross-origin (Hugging Face), but a same-origin mp3
    // must still come out of `audio-v1` with working Range support
    if (isMedia(request, url)) { event.respondWith(cachedOrNetwork(request, url)); return; }
    return;                                   // other same-origin: browser default
  }

  event.respondWith(cachedOrNetwork(request, url));
});

/* exposed purely so the Range parser can be exercised by a test harness */
self.__parseRange = parseRange;

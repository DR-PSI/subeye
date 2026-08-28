/* SubEye service worker
   กฎสำคัญ: บัมพ์ CACHE_VERSION ทุกครั้งที่แก้ index.html
   หน้าเว็บใช้ network-first จึงได้ของใหม่ทันทีที่ออนไลน์ แคชเป็นแค่ตัวสำรองตอนออฟไลน์ */
const CACHE_VERSION = 'subeye-v2.0.3';
const SHELL = ['./', './index.html', './manifest.json'];

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE_VERSION).then(c => c.addAll(SHELL)).catch(()=>{}));
});

self.addEventListener('activate', e => {
  e.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k !== CACHE_VERSION).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('message', e => { if (e.data === 'skipWaiting') self.skipWaiting(); });

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Firebase / API — ไม่แคชเลย
  if (/googleapis|firebaseio|firebaseapp|gstatic\.com\/firebasejs|n8n\./.test(url.hostname + url.pathname)) return;

  const isPage = req.mode === 'navigate' || (req.destination === '' && req.headers.get('accept')?.includes('text/html'));

  if (isPage) {
    // network-first: ได้เวอร์ชันใหม่ทุกครั้งที่ออนไลน์
    e.respondWith((async () => {
      try {
        const fresh = await fetch(req, { cache: 'no-store' });
        const cache = await caches.open(CACHE_VERSION);
        cache.put('./index.html', fresh.clone());
        return fresh;
      } catch {
        return (await caches.match('./index.html')) || Response.error();
      }
    })());
    return;
  }

  // asset อื่น — stale-while-revalidate
  e.respondWith((async () => {
    const cache = await caches.open(CACHE_VERSION);
    const hit = await cache.match(req);
    const net = fetch(req).then(res => { if (res.ok) cache.put(req, res.clone()); return res; }).catch(()=>null);
    return hit || (await net) || Response.error();
  })());
});

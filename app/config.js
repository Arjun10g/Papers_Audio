/* Papers, as Audio — runtime configuration.
 * Bump APP_VERSION together with the APP_VERSION constant at the top of sw.js
 * on every release: the service worker's shell cache name derives from it. */
window.APP_CONFIG = {
  LIBRARY_URL: "https://huggingface.co/datasets/arjun10g/papers-audio/resolve/main/library.json",
  APP_VERSION: "2026.09.16-1"
};

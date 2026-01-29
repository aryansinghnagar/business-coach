(function () {
  const iframe = document.getElementById('appFrame');
  const fallback = document.getElementById('fallback');
  const APP_ORIGIN = 'http://localhost:5000';
  var iframeLoaded = false;

  iframe.addEventListener('error', showFallback);

  // When iframe fires 'load', the app page has loaded. Do NOT access
  // iframe.contentWindow.location or iframe.contentDocument - they throw
  // for cross-origin (localhost), which was incorrectly triggering the fallback.
  iframe.addEventListener('load', function () {
    iframeLoaded = true;
    fallback.style.display = 'none';
    iframe.style.display = 'block';
    sendTabUrlToApp();
  });

  // Send current tab URL to app (for Meet/Zoom/Teams integration)
  function sendTabUrlToApp() {
    if (typeof chrome === 'undefined' || !chrome.tabs || !iframe.contentWindow) return;
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var url = (tabs[0] && tabs[0].url) ? tabs[0].url : '';
      try {
        iframe.contentWindow.postMessage({ type: 'TAB_URL', url: url }, APP_ORIGIN);
      } catch (e) {}
    });
  }

  // Re-send when user switches tabs (panel stays open)
  if (typeof chrome !== 'undefined' && chrome.tabs) {
    chrome.tabs.onActivated.addListener(function () {
      setTimeout(sendTabUrlToApp, 100);
    });
  }

  // Only show fallback if load never fired (e.g. app not running, connection refused).
  // Do NOT check contentDocument/contentWindow.location - they throw for cross-origin.
  setTimeout(function () {
    if (!iframeLoaded) {
      showFallback();
    }
  }, 8000);

  function showFallback() {
    fallback.style.display = 'block';
    iframe.style.display = 'none';
  }
})();

(function () {
  const iframe = document.getElementById('appFrame');
  const fallback = document.getElementById('fallback');
  const APP_ORIGIN = 'http://localhost:5000';
  var iframeLoaded = false;

  iframe.addEventListener('error', showFallback);

  iframe.addEventListener('load', function () {
    iframeLoaded = true;
    fallback.style.display = 'none';
    iframe.style.display = 'block';
    sendTabUrlToApp();
  });

  function sendTabUrlToApp() {
    if (!chrome.tabs || !iframe.contentWindow) return;
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var url = (tabs[0] && tabs[0].url) ? tabs[0].url : '';
      try {
        iframe.contentWindow.postMessage({ type: 'TAB_URL', url: url }, APP_ORIGIN);
      } catch (e) {}
    });
  }

  chrome.tabs.onActivated.addListener(function () {
    setTimeout(sendTabUrlToApp, 100);
  });

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

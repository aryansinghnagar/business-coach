(function () {
  var STORAGE_KEY = 'copilotAppUrl';
  var iframe = document.getElementById('appFrame');
  var fallback = document.getElementById('fallback');
  var setupView = document.getElementById('setup');
  var appView = document.getElementById('appView');
  var setupUrlInput = document.getElementById('setupUrl');
  var setupSaveBtn = document.getElementById('setupSave');
  var changeUrlBtn = document.getElementById('changeUrlBtn');
  var appOrigin = '';
  var iframeLoaded = false;

  function getStorage() {
    return typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync
      ? chrome.storage.sync
      : null;
  }

  function getTabs() {
    return typeof chrome !== 'undefined' && chrome.tabs ? chrome.tabs : null;
  }

  function showView(view) {
    if (setupView) setupView.classList.toggle('active', view === 'setup');
    if (appView) appView.classList.toggle('active', view === 'app');
  }

  function showFallback() {
    if (fallback) fallback.style.display = 'block';
    if (iframe) iframe.style.display = 'none';
  }

  function hideFallback() {
    if (fallback) fallback.style.display = 'none';
    if (iframe) iframe.style.display = 'block';
  }

  function sendTabUrlToApp() {
    var tabs = getTabs();
    if (!tabs || !iframe.contentWindow || !appOrigin) return;
    tabs.query({ active: true, currentWindow: true }, function (tabs) {
      var url = (tabs[0] && tabs[0].url) ? tabs[0].url : '';
      try {
        iframe.contentWindow.postMessage({ type: 'TAB_URL', url: url }, appOrigin);
      } catch (e) {}
    });
  }

  function loadApp(url) {
    if (!url || !iframe) return;
    url = url.replace(/\/+$/, '');
    try {
      appOrigin = new URL(url).origin;
    } catch (e) {
      appOrigin = url.split('/').slice(0, 3).join('/');
    }
    iframeLoaded = false;
    hideFallback();
    iframe.src = url;
  }

  function initFromStorage() {
    var storage = getStorage();
    if (!storage) {
      showView('setup');
      if (setupUrlInput) setupUrlInput.placeholder = 'https://your-copilot-app.example.com';
      return;
    }
    storage.get([STORAGE_KEY], function (data) {
      var url = (data && data[STORAGE_KEY]) ? data[STORAGE_KEY].trim() : '';
      if (!url) {
        showView('setup');
        if (setupUrlInput) setupUrlInput.value = '';
        return;
      }
      if (setupUrlInput) setupUrlInput.value = url;
      showView('app');
      loadApp(url);
    });
  }

  if (iframe) {
    iframe.addEventListener('error', showFallback);
    iframe.addEventListener('load', function () {
      iframeLoaded = true;
      hideFallback();
      sendTabUrlToApp();
    });
  }

  if (setupSaveBtn && setupUrlInput) {
    setupSaveBtn.addEventListener('click', function () {
      var url = (setupUrlInput.value || '').trim();
      if (!url) return;
      if (!url.match(/^https?:\/\//i)) {
        setupUrlInput.focus();
        return;
      }
      url = url.replace(/\/+$/, '');
      var storage = getStorage();
      if (!storage) return;
      storage.set({ [STORAGE_KEY]: url }, function () {
        showView('app');
        loadApp(url);
      });
    });
  }

  if (changeUrlBtn) {
    changeUrlBtn.addEventListener('click', function () {
      showView('setup');
      if (iframe) iframe.src = 'about:blank';
    });
  }

  if (getTabs()) {
    try {
      getTabs().onActivated.addListener(function () {
        setTimeout(sendTabUrlToApp, 100);
      });
    } catch (e) {}
  }

  setTimeout(function () {
    if (!iframeLoaded && appView.classList.contains('active') && iframe.src && iframe.src !== 'about:blank') {
      showFallback();
    }
  }, 10000);

  initFromStorage();
})();

(function () {
  var STORAGE_KEY = 'copilotAppUrl';
  var openTabBtn = document.getElementById('openTab');
  var openSidePanelBtn = document.getElementById('openSidePanel');
  var saveUrlBtn = document.getElementById('saveUrl');
  var appUrlInput = document.getElementById('appUrl');
  var statusEl = document.getElementById('status');

  function setStatus(msg, isError) {
    if (statusEl) {
      statusEl.textContent = msg || '';
      statusEl.className = 'status' + (isError ? ' error' : msg ? ' ok' : '');
    }
  }

  function getStoredUrl(cb) {
    if (typeof browser === 'undefined' || !browser.storage || !browser.storage.sync) {
      if (cb) cb('');
      return;
    }
    browser.storage.sync.get([STORAGE_KEY]).then(function (data) {
      if (cb) cb((data && data[STORAGE_KEY]) ? data[STORAGE_KEY].trim() : '');
    }).catch(function () { if (cb) cb(''); });
  }

  function saveUrl() {
    var url = (appUrlInput && appUrlInput.value) ? appUrlInput.value.trim() : '';
    if (!url) {
      setStatus('Enter an app URL first.', true);
      return;
    }
    if (!url.match(/^https?:\/\//i)) {
      setStatus('URL must start with http:// or https://', true);
      return;
    }
    url = url.replace(/\/+$/, '');
    if (typeof browser === 'undefined' || !browser.storage || !browser.storage.sync) {
      setStatus('Storage not available.', true);
      return;
    }
    var payload = {};
    payload[STORAGE_KEY] = url;
    browser.storage.sync.set(payload).then(function () {
      setStatus('URL saved.', false);
    }).catch(function () { setStatus('Save failed.', true); });
  }

  function openSidePanel() {
    if (typeof browser !== 'undefined' && browser.sidebarAction && typeof browser.sidebarAction.open === 'function') {
      browser.sidebarAction.open()
        .then(function () {
          if (window.close) window.close();
        })
        .catch(function () {
          setStatus('Click "Open in side panel" below.');
        });
      return;
    }
    setStatus('Side panel not available.', true);
  }

  function openTab() {
    getStoredUrl(function (url) {
      if (!url) {
        setStatus('Save an app URL first.', true);
        return;
      }
      browser.tabs.create({ url: url }).then(function () {
        setStatus('Opened in new tab.');
      }).catch(function () { setStatus('Could not open tab.', true); });
    });
  }

  getStoredUrl(function (url) {
    if (appUrlInput) appUrlInput.value = url || '';
  });

  if (saveUrlBtn) saveUrlBtn.addEventListener('click', function () { setStatus(''); saveUrl(); });
  if (openSidePanelBtn) openSidePanelBtn.addEventListener('click', function () { setStatus(''); openSidePanel(); });
  if (openTabBtn) openTabBtn.addEventListener('click', function () { setStatus(''); openTab(); });
})();

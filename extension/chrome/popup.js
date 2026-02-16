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
    if (!chrome.storage || !chrome.storage.sync) {
      if (cb) cb('');
      return;
    }
    chrome.storage.sync.get([STORAGE_KEY], function (data) {
      if (cb) cb((data && data[STORAGE_KEY]) ? data[STORAGE_KEY].trim() : '');
    });
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
    if (!chrome.storage || !chrome.storage.sync) {
      setStatus('Storage not available.', true);
      return;
    }
    chrome.storage.sync.set({ [STORAGE_KEY]: url }, function () {
      setStatus('URL saved.', false);
    });
  }

  function openSidePanel() {
    if (!chrome.sidePanel || !chrome.sidePanel.open) {
      setStatus('Side panel not available.', true);
      return;
    }
    function doOpen(winId) {
      chrome.sidePanel.open({ windowId: winId }).then(function () {
        if (window.close) window.close();
      }).catch(function () {
        setStatus('Click the extension icon to open the side panel.');
      });
    }
    if (chrome.tabs && chrome.sidePanel.setOptions) {
      chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
        if (tabs[0] && tabs[0].id) {
          chrome.sidePanel.setOptions({ tabId: tabs[0].id, enabled: true }).catch(function () {});
        }
        chrome.windows.getCurrent(function (win) {
          if (win && win.id) doOpen(win.id);
          else setStatus('Could not get window.');
        });
      });
    } else {
      chrome.windows.getCurrent(function (win) {
        if (win && win.id) doOpen(win.id);
        else setStatus('Could not get window.');
      });
    }
  }

  function openTab() {
    getStoredUrl(function (url) {
      if (!url) {
        setStatus('Save an app URL first.', true);
        return;
      }
      chrome.tabs.create({ url: url }, function () {
        setStatus('Opened in new tab.');
      });
    });
  }

  getStoredUrl(function (url) {
    if (appUrlInput) appUrlInput.value = url || '';
  });

  if (saveUrlBtn) saveUrlBtn.addEventListener('click', function () { setStatus(''); saveUrl(); });
  if (openSidePanelBtn) openSidePanelBtn.addEventListener('click', function () { setStatus(''); openSidePanel(); });
  if (openTabBtn) openTabBtn.addEventListener('click', function () { setStatus(''); openTab(); });
})();

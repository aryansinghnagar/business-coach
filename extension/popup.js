(function () {
  const APP_URL = 'http://localhost:5000';
  const openTabBtn = document.getElementById('openTab');
  const openSidePanelBtn = document.getElementById('openSidePanel');
  const statusEl = document.getElementById('status');

  function setStatus(msg, isError) {
    if (statusEl) {
      statusEl.textContent = msg || '';
      statusEl.className = 'status' + (isError ? ' error' : msg ? ' ok' : '');
    }
  }

  function openSidePanel() {
    if (typeof chrome === 'undefined' || !chrome.sidePanel || !chrome.sidePanel.open) {
      setStatus('Side panel not available.', true);
      return;
    }
    function doOpen(winId) {
      chrome.sidePanel.open({ windowId: winId }).then(function () {
        if (window.close) window.close();
      }).catch(function () {
        setStatus('Click "Open in side panel" below.');
      });
    }
    // Enable panel for current tab, then open
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

  // Open side panel as soon as popup loads (one click = panel opens)
  if (openSidePanelBtn) {
    openSidePanel();
  }

  if (openSidePanelBtn) {
    openSidePanelBtn.addEventListener('click', function () {
      setStatus('');
      openSidePanel();
    });
  }

  if (openTabBtn) {
    openTabBtn.addEventListener('click', function () {
      setStatus('');
      chrome.tabs.create({ url: APP_URL }, function () {
        setStatus('Opened in new tab.');
      });
    });
  }

  if (openSidePanelBtn && (!chrome.sidePanel || !chrome.sidePanel.open)) {
    openSidePanelBtn.style.display = 'none';
  }
})();

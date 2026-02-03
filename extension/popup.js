(function () {
  const APP_URL = 'http://localhost:5000';
  const openTabBtn = document.getElementById('openTab');
  const openSidePanelBtn = document.getElementById('openSidePanel');
  const statusEl = document.getElementById('status');

  var hasChromeSidePanel = typeof chrome !== 'undefined' && chrome.sidePanel && chrome.sidePanel.open;
  var hasFirefoxSidebar = typeof browser !== 'undefined' && browser.sidebarAction && typeof browser.sidebarAction.open === 'function';

  function setStatus(msg, isError) {
    if (statusEl) {
      statusEl.textContent = msg || '';
      statusEl.className = 'status' + (isError ? ' error' : msg ? ' ok' : '');
    }
  }

  function openSidePanel() {
    if (hasChromeSidePanel) {
      function doOpen(winId) {
        chrome.sidePanel.open({ windowId: winId }).then(function () {
          if (window.close) window.close();
        }).catch(function () {
          setStatus('Click "Open in side panel" below.');
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
      return;
    }
    if (hasFirefoxSidebar) {
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

  if (openSidePanelBtn) {
    openSidePanelBtn.addEventListener('click', function () {
      setStatus('');
      openSidePanel();
    });
  }

  if (openTabBtn) {
    openTabBtn.addEventListener('click', function () {
      setStatus('');
      var api = typeof chrome !== 'undefined' ? chrome : typeof browser !== 'undefined' ? browser : null;
      if (api && api.tabs) {
        api.tabs.create({ url: APP_URL }, function () {
          setStatus('Opened in new tab.');
        });
      }
    });
  }

  // Hide "Open in side panel" only when neither API is available
  if (openSidePanelBtn && !hasChromeSidePanel && !hasFirefoxSidebar) {
    openSidePanelBtn.style.display = 'none';
  }
})();

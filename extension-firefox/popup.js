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
    if (browser.sidebarAction && typeof browser.sidebarAction.open === 'function') {
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
      browser.tabs.create({ url: APP_URL }).then(function () {
        setStatus('Opened in new tab.');
      });
    });
  }
})();

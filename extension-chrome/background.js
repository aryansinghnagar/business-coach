/**
 * Background service worker for Business Meeting Copilot extension (Chrome/Chromium).
 * Opens the side panel when the extension icon is clicked. Uses chrome.sidePanel only.
 */

// Chrome: open side panel when extension icon is clicked (when no popup, or as fallback)
if (chrome.action && chrome.action.onClicked) {
  chrome.action.onClicked.addListener(function (tab) {
    if (!tab) return;
    if (chrome.sidePanel) {
      var tabId = tab.id;
      var windowId = tab.windowId;
      if (chrome.sidePanel.setOptions && tabId) {
        chrome.sidePanel.setOptions({ tabId: tabId, enabled: true }).catch(function () {});
      }
      if (chrome.sidePanel.open && windowId) {
        chrome.sidePanel.open({ windowId: windowId }).catch(function () {});
      }
    }
  });
}

// Chrome 116+: open panel directly on action click when supported
if (chrome.sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(function () {});
}

// Enable side panel for all tabs (required for Meet, Zoom, Teams, etc.)
function enablePanelForTab(tabId) {
  if (!chrome.sidePanel || !chrome.sidePanel.setOptions) return;
  chrome.sidePanel.setOptions({ tabId: tabId, enabled: true }).catch(function () {});
}

function enablePanelForAllTabs() {
  if (!chrome.sidePanel || !chrome.tabs) return;
  chrome.tabs.query({}, function (tabs) {
    tabs.forEach(function (tab) {
      if (tab.id) enablePanelForTab(tab.id);
    });
  });
}

if (chrome.sidePanel && chrome.tabs) {
  enablePanelForAllTabs();
}

chrome.runtime.onInstalled.addListener(function () {
  if (chrome.sidePanel) enablePanelForAllTabs();
});
chrome.runtime.onStartup.addListener(function () {
  if (chrome.sidePanel) enablePanelForAllTabs();
});

if (chrome.tabs) {
  if (chrome.tabs.onCreated) {
    chrome.tabs.onCreated.addListener(function (tab) {
      if (tab && tab.id && chrome.sidePanel) enablePanelForTab(tab.id);
    });
  }
  if (chrome.tabs.onUpdated) {
    chrome.tabs.onUpdated.addListener(function (tabId) {
      if (chrome.sidePanel) enablePanelForTab(tabId);
    });
  }
}

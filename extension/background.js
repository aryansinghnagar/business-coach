/**
 * Background service worker for Business Meeting Copilot extension.
 * Opens the side panel when the extension icon is clicked (no popup, no new tab).
 */

var sidePanel = typeof chrome !== 'undefined' && chrome.sidePanel;

// Open side panel when extension icon is clicked (no popup = onClicked fires)
if (chrome.action && chrome.action.onClicked && sidePanel) {
  chrome.action.onClicked.addListener(function (tab) {
    if (!tab) return;
    var tabId = tab.id;
    var windowId = tab.windowId;
    // Ensure panel is enabled for this tab, then open
    if (chrome.sidePanel.setOptions && tabId) {
      chrome.sidePanel.setOptions({ tabId: tabId, enabled: true }).catch(function () {});
    }
    if (chrome.sidePanel.open && windowId) {
      chrome.sidePanel.open({ windowId: windowId }).catch(function () {});
    }
  });
}
// Also set default behavior so panel opens on action click (Chrome 116+)
if (sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(function () {});
}

// Enable side panel for all tabs (required for Meet, Zoom, Teams, etc.)
function enablePanelForTab(tabId) {
  if (!sidePanel || !chrome.sidePanel.setOptions) return;
  chrome.sidePanel.setOptions({ tabId: tabId, enabled: true }).catch(function () {});
}

function enablePanelForAllTabs() {
  if (!sidePanel || !chrome.tabs) return;
  chrome.tabs.query({}, function (tabs) {
    tabs.forEach(function (tab) {
      if (tab.id) enablePanelForTab(tab.id);
    });
  });
}

// Enable panel for all tabs now and whenever tabs change
if (typeof chrome !== 'undefined' && chrome.tabs) {
  enablePanelForAllTabs();
}
if (typeof chrome !== 'undefined' && chrome.runtime) {
  chrome.runtime.onInstalled.addListener(enablePanelForAllTabs);
  chrome.runtime.onStartup.addListener(enablePanelForAllTabs);
  if (chrome.tabs) {
    if (chrome.tabs.onCreated) {
      chrome.tabs.onCreated.addListener(function (tab) {
        if (tab && tab.id) enablePanelForTab(tab.id);
      });
    }
    if (chrome.tabs.onUpdated) {
      chrome.tabs.onUpdated.addListener(function (tabId) {
        enablePanelForTab(tabId);
      });
    }
  }
}

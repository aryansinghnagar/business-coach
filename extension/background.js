/**
 * Background service worker for Business Meeting Copilot extension.
 * Opens the side panel when the extension icon is clicked.
 * Multi-browser: Chrome/Edge use chrome.sidePanel; Firefox uses browser.sidebarAction.
 */

var sidePanel = typeof chrome !== 'undefined' && chrome.sidePanel;
var firefoxSidebar = typeof browser !== 'undefined' && browser.sidebarAction && typeof browser.sidebarAction.open === 'function';

// Chrome/Edge: open side panel when extension icon is clicked (when no popup, or as fallback)
if (typeof chrome !== 'undefined' && chrome.action && chrome.action.onClicked) {
  chrome.action.onClicked.addListener(function (tab) {
    if (!tab) return;
    if (sidePanel) {
      var tabId = tab.id;
      var windowId = tab.windowId;
      if (chrome.sidePanel.setOptions && tabId) {
        chrome.sidePanel.setOptions({ tabId: tabId, enabled: true }).catch(function () {});
      }
      if (chrome.sidePanel.open && windowId) {
        chrome.sidePanel.open({ windowId: windowId }).catch(function () {});
      }
    } else if (firefoxSidebar) {
      browser.sidebarAction.open().catch(function () {});
    }
  });
}

// Chrome 116+: open panel directly on action click when supported
if (sidePanel && chrome.sidePanel.setPanelBehavior) {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(function () {});
}

// Enable side panel for all tabs (Chrome/Edge only; required for Meet, Zoom, Teams, etc.)
function enablePanelForTab(tabId) {
  if (!sidePanel || !chrome.sidePanel || !chrome.sidePanel.setOptions) return;
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

if (sidePanel && typeof chrome !== 'undefined' && chrome.tabs) {
  enablePanelForAllTabs();
}
if (typeof chrome !== 'undefined' && chrome.runtime) {
  chrome.runtime.onInstalled.addListener(function () {
    if (sidePanel) enablePanelForAllTabs();
  });
  chrome.runtime.onStartup.addListener(function () {
    if (sidePanel) enablePanelForAllTabs();
  });
  if (chrome.tabs) {
    if (chrome.tabs.onCreated) {
      chrome.tabs.onCreated.addListener(function (tab) {
        if (tab && tab.id && sidePanel) enablePanelForTab(tab.id);
      });
    }
    if (chrome.tabs.onUpdated) {
      chrome.tabs.onUpdated.addListener(function (tabId) {
        if (sidePanel) enablePanelForTab(tabId);
      });
    }
  }
}

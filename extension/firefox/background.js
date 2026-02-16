/**
 * Background script for Business Meeting Copilot extension (Firefox).
 * Opens the sidebar when the extension icon is clicked. Uses browser.sidebarAction only.
 */

browser.browserAction.onClicked.addListener(function () {
  browser.sidebarAction.open().catch(function () {});
});

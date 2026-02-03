# Business Meeting Copilot - Browser Extension

This extension provides quick access to the Business Meeting Copilot application from your browser.

## Supported Browsers

- **Chrome** (114+)
- **Edge** (Chromium-based, 114+)
- **Comet** (Chromium-based, 114+)
- **Firefox** (109+)

## Installation

### Chrome / Edge / Comet

1. Open the browser and navigate to `chrome://extensions/` (or `edge://extensions/` / `comet://extensions/`)
2. Enable "Developer mode" (toggle in top-right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The extension icon should appear in your toolbar

### Firefox

1. Open Firefox and navigate to `about:debugging`
2. Click "This Firefox" in the left sidebar
3. Click "Load Temporary Add-on..."
4. Select the `manifest.json` file from the `extension` folder
5. The extension icon should appear in your toolbar

## Usage

1. **Start the application** on your computer:
   ```bash
   python app.py
   ```
   The app will run at `http://localhost:5000`

2. **Open the app**:
   - **Side panel**: Click the extension icon (or use the popup’s **"Open in side panel"** button). The full app runs in the panel; you can be on any site (e.g. a meeting page). Chrome/Edge/Comet use the side panel; Firefox uses the sidebar.
   - **New tab**: Use the popup’s **"Open in new tab"** to open the app at `http://localhost:5000` in a new tab.

## Features

- **Works on any page**: The side panel is enabled for all tabs. Open it on any website; the main page (e.g. Google Meet, Zoom, Teams) stays visible while the app runs in the panel.
- **Video-conferencing integration**: When the current tab is a meeting site (Google Meet, Zoom, Microsoft Teams, Webex, Whereby, GoToMeeting, Jitsi, Discord), the app shows an “Optimized for [name]” badge. The panel opens independently next to the meeting; it does not replace or take over the page.
- **Panel mode**: When the app is loaded inside the extension’s side panel (iframe), it uses a compact layout that fills only the panel, so the sidebar appears independently on other websites.
- **Video feed in sidebar**: The engagement video feed and avatar are in the app’s sidebar, so everything fits in the browser’s side panel.
- **Side panel or new tab**: Open the app in the side panel (icon or popup) or in a new tab (popup). Side panel is enabled for every tab so it works on Meet, Zoom, Teams, etc.

## Troubleshooting

### Side panel doesn't open when clicking the icon

1. Reload the extension: go to `chrome://extensions`, find "Business Meeting Copilot", and click **Reload**.
2. Ensure you're on a normal webpage (e.g. not chrome:// or the extensions page).
3. Click the extension icon in the toolbar again. The side panel should open with the app (or a message to start `python app.py`).

### "App not loading" in side panel

If the side panel shows an error message:
1. Ensure the Flask app is running (`python app.py`)
2. Verify the app is accessible at `http://localhost:5000`
3. Click the extension icon again to refresh the panel

### Extension not appearing

- **Chrome/Edge/Comet**: Check that Developer mode is enabled
- **Firefox**: Extensions loaded via `about:debugging` are temporary and will be removed on browser restart. For permanent installation, package the extension as a `.xpi` file.

### Permission errors

The extension requires permission to access `localhost:5000`. If you see permission errors:
1. Check that `host_permissions` in `manifest.json` includes your app URL
2. Reload the extension after making changes

## Development

### Icons

Icons are in `icons/` (16x16, 48x48, 128x128). Replace them as needed for your branding.

### Manifest Notes

- Chrome/Edge/Comet use `side_panel` key
- Firefox uses `sidebar_action` key (Chrome will ignore this, Firefox will ignore `side_panel`)
- Both browsers will ignore the other's specific keys, so both can coexist in the same manifest

## Files

- `manifest.json` - Extension manifest (MV3)
- `background.js` - Service worker; enables side panel to open on icon click on any page
- `popup.html/css/js` - Extension popup interface
- `sidepanel.html/js` - Side panel/sidebar content (iframe to localhost:5000)
- `icons/` - Extension icons (16x16, 48x48, 128x128)

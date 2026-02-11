# Business Meeting Copilot — browser extensions

Load the Meeting Copilot app in a side panel on any webpage (e.g. Google Meet, Zoom, Teams). **The extension is standalone: it does not require the web app to be running on your machine.** Use a hosted instance of the app instead.

## How it works

1. **Deploy the web app** to a server (e.g. Azure, AWS, or your own host) so it is available at a URL like `https://your-copilot.example.com`.
2. **Install the extension** (Chrome or Firefox, see below).
3. **Set the app URL** in the extension popup (or in the side panel on first open). Save the URL of your hosted Copilot app.
4. **Open the side panel** on any tab. The panel loads your app in an iframe and sends the current tab URL to it (for meeting-site detection). Everything runs in the browser; no local server needed.

## Browser options

| Folder       | Browser               | Install |
|--------------|------------------------|--------|
| **chrome/**  | Chrome, Edge, Chromium | Open `chrome://extensions`, enable Developer mode, **Load unpacked** → select the `chrome` folder. |
| **firefox/** | Firefox 109+          | Open `about:debugging` → This Firefox → **Load Temporary Add-on** → select `firefox/manifest.json`. |

## First-time setup

- Click the extension icon to open the **popup**. Enter your **App URL** (the full URL of your deployed Meeting Copilot app, e.g. `https://your-app.azurewebsites.net`), then click **Save URL**.
- Click **Open in side panel**. The first time, if no URL is saved, the panel shows a setup form; enter the URL there and click **Save and open**.
- To change the URL later, open the popup and edit the App URL, or in the panel click **Change URL** when the app cannot be reached.

## Optional: local development

If you run the app locally (`python app.py`), you can set the App URL to `http://localhost:5000` so the extension loads your local instance. The extension works the same whether the app is local or hosted.

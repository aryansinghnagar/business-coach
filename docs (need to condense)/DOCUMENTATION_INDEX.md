# Documentation Index

Central index for all documentation in the Business Meeting Copilot project. Use this to find the right doc for your role and task.

---

## Getting Started

| Document | Audience | Description |
|----------|----------|-------------|
| [README.md](README.md) | Everyone | Project overview, features, setup, usage, troubleshooting |
| [QUICK_START.md](QUICK_START.md) | Users | Minimal steps to run the app (GUI launcher or CLI) |

---

## Architecture & Configuration

| Document | Audience | Description |
|----------|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Developers | Project structure, design, data flow, extension points |
| [CONFIGURATION.md](CONFIGURATION.md) | Developers / DevOps | All config options and environment variables |
| [API-REFERENCE.md](API-REFERENCE.md) | Developers / Integrators | All HTTP endpoints, request/response shapes |

---

## Running the App

| Document | Audience | Description |
|----------|----------|-------------|
| [LAUNCHER_README.md](LAUNCHER_README.md) | Users | GUI launcher usage, restart/stop options, troubleshooting |

---

## Engagement Detection

| Document | Audience | Description |
|----------|----------|-------------|
| [ENGAGEMENT_DETECTION_DOCUMENTATION.md](ENGAGEMENT_DETECTION_DOCUMENTATION.md) | Users | What engagement detection is, how to use it, interpreting scores |
| [ENGAGEMENT_SYSTEM_README.md](ENGAGEMENT_SYSTEM_README.md) | Developers | Engagement pipeline, components, APIs, face detection backends |
| [EXPRESSION_SIGNIFIERS_DOCUMENTATION.md](EXPRESSION_SIGNIFIERS_DOCUMENTATION.md) | Developers / Analysts | All 30 expression signifiers, formulas, interpretation |
| [ENGAGEMENT_BAR_DOCUMENTATION.md](ENGAGEMENT_BAR_DOCUMENTATION.md) | Users / Developers | Engagement bar UI, behavior, integration |
| [ENGAGEMENT_CONTEXT.md](ENGAGEMENT_CONTEXT.md) | Developers | How engagement context is passed into the AI coach |
| [VIDEO_SOURCE_SELECTION_DOCUMENTATION.md](VIDEO_SOURCE_SELECTION_DOCUMENTATION.md) | Users / Developers | Webcam, file, stream sources; face detection method selection |

---

## Face Detection & Azure

| Document | Audience | Description |
|----------|----------|-------------|
| [AZURE_FACE_API_INTEGRATION.md](AZURE_FACE_API_INTEGRATION.md) | Developers | Azure Face API setup, vs MediaPipe, fallback, configuration |

---

## Debugging & Troubleshooting

| Document | Audience | Description |
|----------|----------|-------------|
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Everyone | Common issues and quick fixes; links to detailed docs |
| [ENGAGEMENT_DEBUG_GUIDE.md](ENGAGEMENT_DEBUG_GUIDE.md) | Developers | Debug endpoints, logs, engagement-detection debugging |

---

## Documentation Layout

```
docs (need to condense)/
├── DOCUMENTATION_INDEX.md              ← You are here
├── README.md                           # Project overview
├── QUICK_START.md                      # Run the app quickly
├── ARCHITECTURE.md                     # Structure and design
├── CONFIGURATION.md                    # Config reference
├── API-REFERENCE.md                    # API reference
├── LAUNCHER_README.md                  # GUI launcher + restart
├── ENGAGEMENT_DETECTION_DOCUMENTATION.md
├── ENGAGEMENT_SYSTEM_README.md
├── EXPRESSION_SIGNIFIERS_DOCUMENTATION.md
├── ENGAGEMENT_BAR_DOCUMENTATION.md
├── ENGAGEMENT_CONTEXT.md
├── VIDEO_SOURCE_SELECTION_DOCUMENTATION.md
├── AZURE_FACE_API_INTEGRATION.md
├── TROUBLESHOOTING.md                 # Common issues and quick fixes
└── ENGAGEMENT_DEBUG_GUIDE.md
```

---

## Quick Links by Role

**End users**  
→ [QUICK_START.md](QUICK_START.md), [README.md](README.md), [ENGAGEMENT_DETECTION_DOCUMENTATION.md](ENGAGEMENT_DETECTION_DOCUMENTATION.md), [LAUNCHER_README.md](LAUNCHER_README.md)

**Developers**  
→ [ARCHITECTURE.md](ARCHITECTURE.md), [API-REFERENCE.md](API-REFERENCE.md), [CONFIGURATION.md](CONFIGURATION.md), [ENGAGEMENT_SYSTEM_README.md](ENGAGEMENT_SYSTEM_README.md), [EXPRESSION_SIGNIFIERS_DOCUMENTATION.md](EXPRESSION_SIGNIFIERS_DOCUMENTATION.md)

**Integrators / DevOps**  
→ [API-REFERENCE.md](API-REFERENCE.md), [CONFIGURATION.md](CONFIGURATION.md), [AZURE_FACE_API_INTEGRATION.md](AZURE_FACE_API_INTEGRATION.md)

**Debugging / Troubleshooting**  
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md), [ENGAGEMENT_DEBUG_GUIDE.md](ENGAGEMENT_DEBUG_GUIDE.md), “Troubleshooting” in [README.md](README.md)

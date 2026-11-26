# SnipVR.IQ

> Built for developers ❤️ who magically become "video editors" every Friday at 5 PM because the client asked for a quick demo.

Trainable Windows desktop app that automatically removes **app initialization / VR shell intro frames** and **ending black fade + outro clips**, keeping only the center clip.

## Features
- **Trainable Intro/Outro Classifier**: Open sample videos in the **Train Model** tab, scrub to mark Start (after init) and End (before black fade), and click **Train & Save Model** (`default_model.npz` is pre-trained on Wellness & Yoga MR).
- **Trim & Export**: Queue single or multiple videos, preview Start/End frames, see per-file **Time Trimmed** and **Trimmed Total Duration**, and export with a Save As filename prompt.
- **1-Tap GitHub Pusher**: Open `Push_To_GitHub_S24_Ultra.html` on mobile or desktop to push the full repo to `https://github.com/xrvnsp/SnipVR.IQ.git` in one atomic commit.

## Quick Start (Windows)
- **Build Standalone `.exe`**: Double-click **`build_portable_bundle.bat`** to generate `dist\SnipVR.IQ.exe` (bundles Python, FFmpeg via `imageio-ffmpeg`, and dependencies).
- **Or Run Directly**: Double-click **`run_app.bat`** if Python 3.11+ is already installed.

## Refresh Release & Mobile Pusher
```bash
python3 rebuild_all.py "your commit message here"
```

---
Made by **SP**

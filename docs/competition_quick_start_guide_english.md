# 🚀 Competition Day Command Quick Start Guide — "More than a Hack 2027"

This quick start guide is designed to help the team run the possession estimation pipeline, perform batch inference with or without an active Internet connection, and launch the interactive 3D viewer on competition day.

---

## 📂 1. Pre-Run Data Placement

Place the official competition datasets (either Metrica Sports split Home/Away CSVs or canonical frame/player tables) inside the project's root data directory:

```powershell
.\data\
```

---

## ⚡ 2. Terminal Commands (PowerShell)

### A. Automatic Batch Processing (Default Offline / Local Mode)

* **When to use:** Closed / air-gapped evaluation environments, no Internet access, or when fast turnaround is critical. Uses the pre-calibrated local System One emulator.
* **Command:**

  ```powershell
  .\.venv\Scripts\python.exe auto_process_competition.py
  ```

* **Expected Output:** Automatically scans `data\`, detects tracking schemas, runs inference across all matches, and exports verified final submission files to `artifacts\submission_*.csv` (`0 NaNs`, contract-validated).

---

### B. Batch Processing with Live Jev AI (Online Mode)

* **When to use:** When outbound network access is available and maximum probabilistic fidelity on contested edge-case duels is desired via live TypeSafe AI servers.
* **Prerequisite:** Set the API key environment variable in your session:

  ```powershell
  $env:TYPESAFE_API_KEY = "your_api_key_here"
  ```

* **Command:**

  ```powershell
  .\.venv\Scripts\python.exe auto_process_competition.py --live-jev
  ```

---

### C. Update & Build the Interactive 3D Broadcast Viewer

* **When to use:** After generating match predictions, to update the FC Barcelona-themed 3D replay interface for visual sanity checks or team demonstrations.
* **Command:**

  ```powershell
  .\.venv\Scripts\python.exe generate_3d_replay_html.py --standalone
  ```

* **Expected Output:** Updates `artifacts\visualizations\match_payload.js` (preserving hot-edits on `viewer.html`) and generates the portable bundle `artifacts\visualizations\barca_3d_interactive_replay.html`.

---

## 🎮 3. 3D Viewer Key Controls (`viewer.html`)

Open `artifacts\visualizations\viewer.html` in your browser and use the following keyboard shortcuts and UI controls:

| Key / Control | Viewer Action |
| :--- | :--- |
| **`Space`** | Play or pause match animation. |
| **`Keys 1, 2, 3, 4`** | Switch camera perspective: **1: TV**, **2: Drone**, **3: Top-Down Tactical**, **4: Free Orbit** (click-drag to rotate, scroll wheel to zoom). |
| **`Key M`** | Toggle live comparison: **`🧠 OUR AI (Barça + Viterbi + Jev)`** vs. **`🤖 NAIVE AI (Proximity Baseline)`** to visually demonstrate false turnover mitigation. |
| **`Key H` / `👁️ Vista`** | Toggle **Cinema / Clean View** mode to hide lateral HUD cards and maximize pitch visibility. |
| **`Timeline Pips`** | Click colorful vertical tick markers along the timeline slider track to jump directly to turnovers or Jev AI decision points. |

---

## ✅ 4. Pre-Submission Checklist

Before uploading any artifact to the evaluation platform, confirm that:

1. The terminal output confirms: `[PASS] Submission CSV written and contract-validated` with `0 NaNs`.
2. Global possession share (Home % vs. Away %) reflects realistic soccer match distributions.
3. The expected `submission_*.csv` files are present and non-empty in the `artifacts\` directory.
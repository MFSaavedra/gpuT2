# gol_gui — Tarea 3 demo video storyboard (silent capture → separate ES/EN explainer captions)

Target length **~3:55**, 29 cues (requirement: 3–4 min). Record silent; the captions are an
**explainer track** — they teach the implementation, not just label the sections. After you record,
send me the real per-scene durations and I'll re-time the cues to match your footage.

**The video opens on the headless `gol` (terminal), then moves to the `gol_gui` viewer.**

**Recording tips**
- 1080p, ~30 fps. Captions are single-line (one language track at a time) but carry real detail —
  move slowly and hold each shot long enough to read.
- **Opening (0:00–0:14), headless `gol` with the terminal renderer** — quick "same simulation, no GUI"
  shot. In-place ANSI animation reads best on camera; the scrolling text dump is the literal `text`:
  ```bash
  ./build/gol --engine cpu --renderer ansi -r 40 -c 80 -g 300 --rle patterns/highlife_replicator.rle
  #   in-place animation; arrow keys pan, space pauses, q quits. Use --renderer text for the scroll dump.
  ```
- Then launch the viewer with `scripts/run_gui.sh 1024x1024 --rle patterns/highlife_replicator.rle`
  so the zero-copy path is used. Hold the **terminal** on camera for a beat (`GL_RENDERER = NVIDIA…`)
  and keep the **window title** ("Game of Life — CudaEngine / zero-copy interop") in frame — that pair
  is your interop proof (cues 8–9).
- Several scenes explain code. For those, a **1–2 s cutaway to the source file in an editor** makes
  the caption land; the "cutaway" column says which file. Keep cutaways short — the app is the star.
- For the interactivity scene (H), run **`showmethekey-gtk`** (reads evdev globally, so it catches
  keys sent to the XWayland `gol_gui` window; first launch asks for a pkexec password). Drag its
  always-on-top overlay **inside your capture region** so keystrokes show on camera.

| Scene | Window | Cues | On screen — what to do | Cutaway (optional) |
|-------|--------|------|------------------------|--------------------|
| A. Headless `gol` (terminal) | 0:00–0:14 | 1–2 | Run the ANSI/text command above; a few gens of HighLife animating in the terminal. | terminal |
| B. Architecture + pivot | 0:14–0:48 | 3–6 | Launch `gol_gui`; board running while captions explain the design and the ping-pong. | `ISimEngine.hpp`; UML `report/img/uml_architecture.png` |
| C. Launch + interop proof | 0:48–1:10 | 7–9 | Show `run_gui.sh` + `GL_RENDERER = NVIDIA…`; hover the title bar's "zero-copy interop". | terminal |
| D. Interop mechanism | 1:10–1:52.5 | 10–14 | 1024² board evolving smoothly — the payoff of zero-copy. | `src/render/CudaGlInterop.cu` (register + D2D copy) |
| E. Shader pipeline | 1:52.5–2:25.5 | 15–18 | App running; optionally flash the shaders. | `src/gui/shaders/display.vert` + `display.frag` |
| F. Colour modes | 2:25.5–2:58.5 | 19–22 | Qt "Colour" dropdown: Binary → Neighbour count → Age heatmap, pausing on each. | — |
| G. Palettes + grid lines | 2:58.5–3:15 | 23–24 | In Age heatmap, switch "Palette" to **Spectrum** (age by hue); then wheel-zoom until grid lines appear. | — |
| H. Interactivity | 3:15–3:39.5 | 25–27 | Wheel-zoom (cell under cursor stays fixed), pan, left-drag paint, Shift-erase; press S to step. | keystroke overlay: `showmethekey-gtk` |
| I. Verify + outro | 3:39.5–3:55 | 28–29 | "Open RLE…" → `highlife_replicator.rle` running; end on the board + title bar. | — |

Captions are **two separate single-language tracks** (never both on screen at once):
- `captions.es.srt` — Spanish
- `captions.en.srt` — English
- `captions.bilingual.srt` — authoring master (both lines); the es/en files are split from it.

Burn one with `./burn_captions.sh demo.mp4 es` (or `en`), or mux both as toggleable soft-sub
tracks (see the script's footer). Technical claims are pulled from the real code
(`Config.cpp`, `GolGlWidget.cpp`, `MainWindow.cpp`, `CudaGlInterop.cu`, `display.vert/.frag`).

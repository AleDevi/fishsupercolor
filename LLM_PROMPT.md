# LLM prompt: working on Guppy Colour with an AI assistant

**For the researcher.** Before you ask an LLM (Claude, ChatGPT, Gemini, a coding agent…) to help with this
repository, paste this whole file as your first message, then say what you want to do. Coding agents that read
`AGENTS.md` or `CLAUDE.md` load it by themselves. The part between the `AUTO` markers is rewritten from the code
on every push to `main`; the rest is kept up to date by hand with each change (see
[Keeping this file current](#keeping-this-file-current)).

---

You are helping a researcher work on **Guppy Colour** (`guppycolor`), an open-source desktop app that measures
the colour pattern of fish from photos. Read this whole text before answering. It tells you what the project is
for, how it is built, the rules every change must follow, and how to organise the work with the researcher.

## 1. Goal and users

- **Goal**: measure the morphology and colouration of fish, automatically and reproducibly
  from photos, replacing slow and subjective hand measurement in ImageJ: size (SL, TL, depth, areas), the area
  and number of spots of each colour class, and the spatial pattern (to compare individuals and find groups of
  similar liveries). The tool is optimized for measuring colour pattern of male guppies (*Poecilia reticulata*). 
- **Users**: behavioural ecologists and evolutionary biologists. They use **R/RStudio**, often do **not** know
  Python, and work on Windows PCs (some lab PCs have an NVIDIA GPU, most do not). So:
  - everything happens in a **GUI**; installation is **one double-click** (`install_windows.bat`), no terminal;
  - results are **CSV files ready for `read.csv()` in R** (comma, decimal point, `NA`, `TRUE`/`FALSE`);
  - messages in the app are plain, short, non-technical English and say what to do next.
- **Scope**: one fish per photo, batches of hundreds of photos per experiment, fish anaesthetised and lying on
  their side (head usually on the left), on graph paper, with or without a ColorChecker 24 chart and a written or
  QR label with the fish id.
- **Not only male guppies.** The app is *optimised* for male guppies, the first use case, but it is and will be
  designed for other animals: guppy females, other fish species, and possibly other animals. Photo sets may
  already mix males, females and other species. So:
  - everything specific to a species or sex (body parts, colour classes, landmarks, expected shapes, label
    formats) lives in a **profile** (`guppycolor/profiles/*.json`), never in the code;
  - no guppy-only or male-only assumption may be hard-coded (e.g. a gonopodium, three colour classes, a fixed
    length/depth ratio, the head on the left): read it from the profile or the settings, with the guppy male as
    the default;
  - when a feature only makes sense for some animals, make it optional and say in the profile when it applies;
  - name things generically in code, CSV columns and UI (`fish`, `animal`, `colour class`, `landmark`), keeping
    the guppy names only as profile values.

## 2. Scientific context

- Guppy males carry three kinds of colour, the classes of the default profile: **carotenoid** (yellow, orange,
  red; pigment from the diet, sexually selected), **black** (melanin) and **iridescent / structural** (blue,
  violet, white, green). Classes may be split more finely later.
- **Reference method: van der Bijl et al. 2025, *Nature Ecology & Evolution*** (deep learning on thousands of
  standardised guppy photos; code: `github.com/Ax3man/vdBijl_etal_2025_GuppyColorPatterns`, CC BY 4.0; photos on
  FRDR, doi 10.20383/103.01281). This app follows its recipe: fish mask, colour-class masks, landmarks plus
  outline semi-landmarks, generalised Procrustes alignment and thin-plate-spline warp of every fish onto the mean
  shape, comparison on the aligned maps. Their trained models are not public, so the app reimplements the steps
  in Python and trains its own models on the user's photos.
- Other methods it draws on: Menesatti et al. 2012 (per-image 3D thin-plate-spline colour correction with a
  ColorChecker), Van Belleghem et al. 2018 (*patternize*, pattern alignment in R), Valvo et al. 2021
  (*Colormesh*), ilastik (random forest on hand-made pixel features trained from brush strokes).
- What counts as a good measure: agreement with hand measures on the same fish (the app reports bias, mean
  absolute error, correlation, Bland-Altman limits) and, for patterns, recovering known groups (adjusted Rand
  index). Colour comparisons across sessions are valid only with a colour chart in the photo; photos without one
  are measured uncorrected and flagged. Any change to how something is measured is a change to the science:
  it must be explained, tested and written in the CHANGELOG, and old projects must say how to get the new values.

## 3. How the app works

- **Pipeline for one photo** (`pipeline.py`): colour correction (ColorChecker found automatically, 3D TPS to
  nominal values or to a reference photo) → fish outline (YOLOE with the text prompt "fish" → SAM 2.1; a click in
  Review prompts SAM instead) → scale (graph-paper grid by autocorrelation, or a line drawn by hand) → sizes
  (outline first, landmarks win where they exist: `sizes.py`) → fish id (QR, else OCR constrained to expected
  formats, or file name, or a data table) → colour classes (random forest trained in the app) → quality flags.
- **Pixels until the export**: results are stored in pixels per photo (`results/<photo>.json`); conversion to mm
  happens in `export.py`, so a doubtful scale can fall back to the session median without a new analysis.
- **Project** = one experiment = one folder (`project.json`, results, masks, models, exports). Photo folders
  (sessions) are added over time; only new or changed photos are analysed. Photos are never modified.
- **Models trained in the app**: colour classes (random forest stored as plain `.npz` arrays, evaluated with
  numpy) and landmarks (YOLO-pose fine-tuned from `yolo11n-pose.pt`). Both can be exported/imported as a `.zip`
  with a SHA-256 manifest.
- **Layers**: `guppycolor/engine/` = stateless algorithms (plain functions, no project or GUI state);
  `guppycolor/*.py` = project, pipeline, export, training, landmarks, pattern; `guppycolor/gui/` = PySide6
  window, one screen per step. Keep that separation: engine code must stay testable without a window.
- The full user documentation (screens, every CSV column, every flag, project folder layout) is in `README.md`;
  the history of every change is in `CHANGELOG.md`. Read the parts you need before changing behaviour.

<!-- AUTO:START (generated by tools/update_llm_prompt.py: do not edit by hand) -->

### Current state (generated)

- Version: **0.12.0** (`guppycolor/__init__.py` and `pyproject.toml`).
- Target Python: **3.12** on Windows (tests and installer). Pinned in `requirements.in`: `ultralytics==8.4.183`.
- App screens (README, Workflow): 1 Project; 2 Settings; 3 Analysis; 4 Review; 5 Results; 6 Pattern and clusters; 7 Training (colour classes); 8 Landmarks.

#### Recent versions (CHANGELOG)

- **Unreleased**: New `training_kit/` (separate from the app, not in its release zip): standalone…
- **0.12.0 (2026-10-27)**: Optional colour classes: the iridescent class is now optional (`"optional": true` in the…; Ritocco for colour models: Training > Start from: a colour model that keeps its training…; Import model (colours) accepts a model with only some of the project's classes (e.g.…; New guide `MODELS_AND_TRAINING.md`: which models exist, where they are saved, which file…
- **0.11.0 (2026-10-26)**: Outline models: Settings > Fish outline can use an imported outline model instead of the…; Ritocco: Landmarks > Start from: training can start from a landmark model of the project…; Landmark models can say which landmarks they were never trained on (the training kit's…; Export / Import model works for outline models too.
- **0.10.0 (2026-10-23)**: Landmarks win over the outline (new module `sizes.py`), as soon as they are clicked on…; Review shows each photo in separate views, Fish (outline and measures), Label (the label…; Add to the outline is now a click mode in the Fish view ("A click on the fish: redoes the…
- **0.9.0 (2026-10-23)**: Results: the table now has `standard_length_mm` and `total_length_mm`; Fixed: on photos with the fish upright (head up or down) the "head on the left/right"…; Training and Landmarks screens: the crop around the fish leaves more room at the head and…

#### Planned (CHANGELOG)

- General scope (Ale, 2026-10-23): tuned for male guppies, built for any fish, both sexes and later other
  animals. Next: animal groups in one project (e.g. male / female, or species), each with its own profile and
  landmarks, assigned by folder, file name or per photo; a female guppy profile (anal fin base instead of the
  gonopodium for body depth); a generic fish profile.
- Python 3.13: make rapidocr-onnxruntime optional (no 3.13 release; install fails), installer should `venv --clear`.
- Fish detector: YOLOE finds no fish on 36/50 white-background photos (Dick 2012-13, datasets/test-100-2026-10-22): add a fallback (background threshold + SAM) or train the detector.
- Body-part model with its annotation screen.
- Ritocco for outline models (needs outlines split into body and caudal fin on the researcher's photos) and
  mixing a few van der Bijl photos into in-app training, as the training kit suggests.
- Spectra estimated from the photos, as a separate long CSV (default reference library ['guppy_orange_cw'], standard
  colorchecker_simulated_colourworker). Run only when the chart fit uses all 24 patches with error <= 0.1, otherwise
  flag the photo (chart incomplete) and leave its spectra empty; spot masks only inside the fish mask; ~0.3 s/photo.
  Test report: guppy-tool/spettri/test_spettri_2026-10-24.md.
- Scale from a ruler in the photo (not only graph paper).
- RAW photo support.
- Verified update from GitHub releases (off while the repository is private).
- Landmark model export as plain tensors (safetensors/npz) rebuilt locally on import, so a shared file never goes through an unpickler (security review suggestion).

#### Species profiles

- `guppy_male.json` (Poecilia reticulata, male)
  - parts: body, caudal, dorsal, gonopodium
  - colour classes: orange, black, iridescent
  - landmarks, in clicking order: snout, eye (optional), dorsal_base, gonopodium_base (optional), peduncle_end, tail_tip (optional)

#### Modules (first line of each docstring)

| File | What it does |
| --- | --- |
| `guppycolor/__init__.py` | Guppy Colour: batch measurement of fish colour patterns |
| `guppycolor/__main__.py` | python -m guppycolor -> opens the window python -m guppycolor run PROJECT -> analyses pending photos and writes export/measurements.csv (no window) |
| `guppycolor/export.py` | CSV export, one row per fish, ready for R (comma separator, decimal point, NA for missing values) |
| `guppycolor/labelsheet.py` | Printable label sheet: one label per fish with a QR code + the same text in large print, so engine/labels.py reads the QR (no errors) and a person (or the OCR… |
| `guppycolor/landmarks.py` | Named landmarks (from the species profile): clicked by hand in screen 8, or predicted by a YOLO-pose model trained on those clicks |
| `guppycolor/metadata.py` | Photo metadata: fields parsed from the file name with a user pattern, plus EXIF |
| `guppycolor/modelshare.py` | Export a trained model to one .zip file and import it into another project (or another computer) |
| `guppycolor/outline.py` | Outline models: a YOLO segmentation model that draws the parts of the fish (e.g. body and caudal fin) on the whole photo, imported as a .zip (Export model /… |
| `guppycolor/pattern.py` | Pattern and clusters (screen 6): every analysed fish with a colour map is aligned on the mean shape (engine/pattern.py), described as per-pixel colour maps and… |
| `guppycolor/pipeline.py` | Analysis of one photo: colour correction -> fish mask -> scale -> measures -> quality flags |
| `guppycolor/profile.py` | Species profile: body parts, colour classes and named landmarks |
| `guppycolor/project.py` | A project = one experiment: a folder holding settings, one result file per photo, masks and exports |
| `guppycolor/scaleline.py` | Scale set by hand: a line drawn on a photo over a known length (ruler, graph paper, any object) and that length in mm. scale/<photo id>.json holds the line of… |
| `guppycolor/sizes.py` | Sizes of one fish: from the outline first, then the landmarks win where they exist |
| `guppycolor/training.py` | Colour-class training inside a project: brush strokes per photo -> random forest -> areas per class |
| `guppycolor/verify.py` | Check the app against measures made by hand (one row per fish: total area per colour class in mm², total length, area) |
| `guppycolor/engine/__init__.py` | Image-analysis algorithms (colour chart, segmentation, scale, morphometry) |
| `guppycolor/engine/colour.py` | ColorChecker Classic 24 detection + per-image TPS-3D colour correction (Menesatti et al. 2012) |
| `guppycolor/engine/labels.py` | Text written in the photo: QR code first, else handwritten/printed labels via OCR |
| `guppycolor/engine/morpho.py` | Standard length, total length and areas from a fish mask (pixels |
| `guppycolor/engine/pattern.py` | Colour pattern of many fish on one common shape (phase 3) |
| `guppycolor/engine/pixels.py` | Colour classes per pixel, ilastik-style: hand-made features (Lab colour smoothed at a few scales, edges, blob filters) + a random forest trained on the user's… |
| `guppycolor/engine/scale.py` | Pixels per mm from the graph paper around the fish (smallest grid square = 1 mm) |
| `guppycolor/engine/segment.py` | Zero-training fish segmentation: YOLOE (text prompt 'fish') -> box -> SAM 2.1 mask |
| `guppycolor/gui/calibrate.py` | Calibrate manually: the whole photo with zoom (wheel) and pan (right button) |
| `guppycolor/gui/landmarks_page.py` | Screen 8, Landmarks: click the profile's named landmarks in order on a photo (snout, eye, dorsal base…) |
| `guppycolor/gui/main.py` | Main window: one screen per step (Project, Settings, Analysis, Review, Results, Pattern and clusters, Training, Landmarks) |
| `guppycolor/gui/modelfiles.py` | Export / Import model buttons shared by the Training and Landmarks screens |
| `guppycolor/gui/pattern_page.py` | Screen 6, Pattern and clusters: align every fish on the mean shape, compare the colour patterns with the chosen distance, group them with a similarity slider |
| `guppycolor/gui/training_page.py` | Screen 7, Training: paint examples of each colour class on a few fish, train, use the model, check it against areas measured by hand |

#### Tests

- `test_app.py` (20), `test_gui.py` (5), `test_labels.py` (8), `test_landmarks.py` (17), `test_modelshare.py` (3), `test_outline.py` (5), `test_pattern.py` (8), `test_training.py` (11), `test_training_kit.py` (10).

<!-- AUTO:END -->

## 4. Rules for every change

1. **English everywhere** in the repository: UI texts, CSV columns, code, comments, docs. (The researcher may
   talk to you in another language; answer in theirs, write the repository in English.)
2. **CHANGELOG.md**: every user-visible change gets a line under a new version heading
   `## X.Y.Z (YYYY-MM-DD)`, written for the user (what changes for them, what to do, e.g. "run the Analysis
   again"). Planned work goes under `## Planned`.
3. **README.md**: update the sections the change touches (screens, CSV columns, flags, project folder, security).
   **MODELS_AND_TRAINING.md**: update it with every change to models, training, import/export or the GuppyTraining
   kit, and its "last update" line at every release: it is the researcher's single guide to where models are and
   how to train and load them.
4. **Version**: bump it in both `guppycolor/__init__.py` and `pyproject.toml` (semantic: minor for features,
   patch for fixes).
5. **CSV for R**: never rename or remove a column silently; add new columns, keep `NA` for missing values, and
   add a `*_source` column when a value can come from different methods.
6. **Tests**: add or update tests in `tests/` (synthetic photos from `tests/conftest.py` and
   `tests/synthetic_fish.py`; tests must not download models). Run `python -m pytest tests` before committing.
7. **Security invariants** (the app runs on institute PCs; do not weaken these without the researcher's explicit
   decision, and record it in the CHANGELOG):
   - dependencies only from the hash-locked `requirements-win.txt` (binary wheels only); to change them, edit
     `requirements.in` and recompile the lock (command in README, For developers);
   - every downloaded model is checked against a fixed SHA-256 before it is opened; torch weights are loaded
     weights-only; colour models never go through pickle;
   - no telemetry, no automatic package installs, no network access except the model downloads listed in the code;
   - names read from project files are restricted to `[A-Za-z0-9_-]`; text Excel would read as a formula is
     prefixed with `'` in CSVs; nothing is written outside the project folder, `%LOCALAPPDATA%\GuppyColour` and
     `~/.guppycolor`; no administrator rights.
8. **Style**: match the surrounding code (short docstrings that say what and why, few comments, plain functions).
   Every module starts with a docstring: its first line is copied into this file's module table.
9. **Generic by design**: a change must work for any profile, not only the guppy male (section 1, *Not only
   male guppies*). If it needs a new species- or sex-specific value, add it to the profile with the current
   guppy value as the default, and add a test with a profile that lacks it.
10. **This file**: update the hand-written sections when you change the goal, architecture, rules or way of
   working, and run `python tools/update_llm_prompt.py` (see below).

## 5. Build, test, release

- **Run from source** (developers): Python 3.12, `pip install --require-hashes --only-binary=:all: -r
  requirements-win.txt`, then `pip install --no-deps -e .`; `python -m guppycolor` opens the window,
  `python -m guppycolor run PROJECT` analyses without a window.
- **Tests**: `python -m pytest tests`. GitHub Actions (`.github/workflows/tests.yml`) runs them on Windows with
  Python 3.12 and the locked packages on every push and pull request.
- **Release** (done by the maintainer, by hand):
  1. version bumped, CHANGELOG and README updated, tests green;
  2. **security re-check** of the exact zip that users will install (lock hashes against PyPI, model hashes, any
     new file access, network use or deserialisation). No release without it;
  3. push the tag `v<version>`: `.github/workflows/release.yml` builds `GuppyColour_<version>.zip` with
     `git archive` plus its `.sha256` and publishes the release with the CHANGELOG section as notes.
- Users update by unzipping the new version and running `install_windows.bat` again; projects are kept.

## 6. How to work with the researcher

- **Explain in plain words.** The researcher knows the biology very well and may not read Python. Say what a
  change does to their measures and screens, not only what you changed in the code. Avoid jargon or explain it.
- **Ask before deciding science.** How a length, area or colour class is defined, which photos are excluded, how
  fish are matched or aligned: propose options with their consequences and let the researcher choose. Technical
  choices that can be undone (a refactor, a test, a branch) you can make yourself and report.
- **Small steps, each one usable.** One feature or fix per change, with its version, CHANGELOG line and tests.
  Prefer a working simple version now over a complete one later.
- **Check on real photos when you can.** Synthetic tests prove the code runs; only real photos (the researcher's,
  or public datasets such as van der Bijl 2025 and Dick et al. 2018) show whether a measure is right. Ask for a
  few example photos when a change depends on how real photos look. Never commit photos or personal data.
- **Say what you did not check.** If you could not run the tests or the GUI, say so.
- **Before you finish**, list what the researcher has to do (try a screen, run the Analysis again, re-check
  security before a release) and propose any update this file needs.

## 7. How to organise the chats

Long chats lose track. Split the work like this:

- **One planning chat** for the project as a whole: goals, priorities, decisions, the roadmap. It does not write
  code; it opens task chats and collects their results.
- **One chat per task or topic**, each started by pasting this file plus a short task description, for example:
  app screens and installer; one algorithm (segmentation, scale, colour correction, labels/OCR, landmarks,
  pattern and clusters); training a model on a dataset; literature; the security re-check of a release (a
  separate chat that did not write the code it checks).
- **Ownership**: only one chat at a time edits a given file or module, to avoid conflicting changes. A task chat
  that needs a change elsewhere writes it down for the owner instead of editing it.
- **Hand-off note** at the end of every task chat: what was done, files changed, version, tests run, decisions
  taken and by whom, what is still open. Paste it into the planning chat. Lasting decisions go into the CHANGELOG
  (or into this file, if they change how the project works) so the next chat finds them in the repository.
- **Start a new chat** when one gets long or changes topic: paste this file and the last hand-off note.

## Keeping this file current

- The block between the `AUTO` markers is generated by `tools/update_llm_prompt.py` from `guppycolor/__init__.py`,
  the module docstrings, `tests/`, the species profiles, `requirements.in`, the README screens and the CHANGELOG.
  Never edit it by hand: run `python tools/update_llm_prompt.py`.
- GitHub Actions (`.github/workflows/llm-prompt.yml`) checks it on every pull request and, after every push to
  `main`, regenerates it and commits the result if anything changed.
- The hand-written sections are updated with the change that makes them wrong (rule 10 above). When you finish a
  task, re-read sections 1–7 and propose edits if anything no longer holds.

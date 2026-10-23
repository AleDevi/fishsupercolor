# Changelog

## Planned
- Python 3.13: make rapidocr-onnxruntime optional (no 3.13 release; install fails), installer should `venv --clear`.
- Fish detector: YOLOE finds no fish on 36/50 white-background photos (Dick 2012-13, datasets/test-100-2026-10-22): add a fallback (background threshold + SAM) or train the detector.
- Body-part model with its annotation screen.
- Spectra estimated from the photos, as a separate long CSV.
- Scale from a ruler in the photo (not only graph paper).
- RAW photo support.
- Verified update from GitHub releases (off while the repository is private).

## 0.6.1 (2026-10-23)
- Installer updates an installation it made itself (marker file `venv\guppycolour-installed.txt`) in place (packages
  already installed are not downloaded again; any other venv is replaced),
  keeps no pip cache (about 1 GB less on disk), hides pip's "new version" notice, and accepts only Python 3.12
  again (3.13 fails on `rapidocr-onnxruntime`). README: what can be deleted to free disk space.
- GitHub Actions: tests on Windows/Python 3.12 at every push, releases from a `v<version>` tag.
- Tests: `tests/data/samples/` with one real photo per photo setup (8 photos, neutral ids, README and CSV);
  not included in the release zip.

## 0.6.0 (2026-10-22)
- Scale: the automatic graph-paper scale must be measured over at least 20 squares (2 cm), else it is doubtful.
  A 2 cm line along the grid is kept (`scale_line`, `scale_squares` in the results).
- Review: a doubtful scale shows the whole photo with the red 2 cm line: **Confirm scale**, or **Calibrate
  manually** (zoomable window: click the two ends of a known length, type it in mm, optionally for the whole folder).
- Settings: scale **by hand** (every photo goes to Review until a line is drawn for it or its folder; flag
  `scale_to_set`) and **Measure on a photo…** to draw the fixed px/mm value.
- Scale drawn by hand is kept in `scale/` and wins over the automatic one in later analyses; new `scale_source`
  values `line` and `line_folder`.
- Review: after a click redoes the outline, a message says what still needs a look (e.g. a doubtful scale).
- Review: an outline from a click that is much larger than the other fish (background included) is not saved
  without asking.

## 0.5.1 (2026-10-22)
- Installer accepts Python 3.13 as well as 3.12 (same locked packages, binary wheels for both), looks in the usual python.org install folders, and says which Python it uses.
- Landmark training sets `amp=False`, so on a GPU ultralytics never downloads `yolo11n.pt` for its AMP check (security review note).

## 0.5.0 (2026-10-22)
- Screen **8 Landmarks**: click the profile's named landmarks (snout, eye, dorsal fin base, gonopodium base,
  end of the caudal peduncle, tip of the tail) in order; S skips, Backspace undoes, N next photo.
- **Standard length**: `standard_length_mm` (snout to end of the peduncle) from the landmarks, updated as soon as
  they are clicked. New CSV column `landmarks_source` (`manual`, `model`).
- Landmark model: YOLO-pose fine-tuned in the app on the clicks (`models/landmarks_vN`), from `yolo11n-pose.pt`
  checked against its SHA-256; the trained weights are hash-checked before every use. Clicked landmarks win.
- The pattern screen uses the landmarks (snout for the head, Procrustes landmarks in the alignment).
- New flags `sl_landmarks_missing` and `landmark_model_error`.

## 0.4.0 (2026-10-22)
- Screen 6 "Pattern and clusters". Every analysed fish with colour classes is aligned on the mean body shape: outline semi-landmarks (snout, peduncle, 10 dorsal + 10 ventral points, plus any named Procrustes landmarks present in every fish) → generalised Procrustes analysis → thin-plate-spline warp of the colour map onto the mean shape.
- Two descriptions from the same aligned maps: per-pixel colour maps, and spots as points (round) or lines (length/width ≥ 2.5, adjustable).
- Choice of distance: colour maps (weighted Jaccard on the aligned maps, with a tolerance blur), spots as points and lines (one-to-one matching), colour amounts only (Bray-Curtis). Colour classes to compare can be ticked on/off.
- Hierarchical clustering (average, complete or single linkage) with a similarity slider; the tree, the clusters and their mean colour maps update live as the slider moves. Shows the cophenetic correlation and, when labels carry groups, the agreement with them (adjusted Rand index).
- Settings: head side in the photos (left/right/guess; a snout landmark wins), front of the head left out (default 0.13 body lengths, so the eye is not a black spot), region (body or whole fish).
- Export to `export/pattern/`: `pattern_fish.csv` (cluster, % of each colour on the aligned body, colour centre, number of points/lines), `pattern_spots.csv` (one row per spot: kind, position, ends, length, width, angle, area), `pattern_grid.csv` (colour fraction per grid cell, long format), `pattern_shape.csv` (Procrustes coordinates, for geomorph), `pattern_distances.csv` (distance matrix), `pattern_settings.csv`, `cluster_NN.png`.
- The pipeline keeps each photo's colour class map (`masks/<photo_id>_colours.png`); older results get it rebuilt on first use.
- What to measure: "size + colour" or "size only" (e.g. females), set for the project, per folder (Project screen), by a pattern on the id or file name, or per photo in Review.
  Size-only fish get NA in the colour columns. New CSV column `analysis` (`size` or `size_colour`).
- Setting **Head of the fish** (left by default, right, or guessed), used by the measures and the pattern screen.
- Comparison with measures made by hand also covers total length and fish area, works without a colour model,
  and recognises total columns (e.g. `OrangeT`, `BlackT`, `Totallength`, `Area`).
- Name check uses a full match (security review nit).

## 0.3.0 (2026-10-22)
- The whole app is in English: screens, messages, CSV columns, status values, warnings, project files, installer and docs.
  Projects made with 0.1–0.2 must be created again (results are recomputed anyway).
  Main renames: `misure.csv` → `measurements.csv`, `progetto.json` → `project.json`, `stato` → `status`,
  `avvisi` → `flags`, `etichetta_*` → `label_*`, colour classes `orange`, `black`, `iridescent`.
- Explicit **Fish ID** source in Settings: none, label in the photo, file name `{id}`, or data table + shooting
  sequence (from the label-reading module). Optional label reading for date and group.
- `total_length_mm`: TL when the whole caudal fin is in the outline, else NA with flag `tail_incomplete`.
- Image quality per fish: `fish_brightness`, `saturated_pct`, `sharpness` columns; flags `overexposed` and `blurry`.
- Flag `date_mismatch`: date written on the label differs from the EXIF date.
- Profile and model names from a project file are validated (security review).
- `README.md` rewritten in full, this changelog, `LICENSE` (AGPL-3.0).
- Installer files renamed `install_windows.bat` and `start_guppycolour.bat`.

## 0.2.0 (2026-10-22)
- Screen **7 Training**: brushes generated from the species profile (one per colour class + body without colour
  + light reflection, keys 1–9, eraser, zoom), strokes saved per photo, random-forest pixel classifier,
  versioned models, prediction overlay, "most uncertain first", import of strokes painted in Windows Paint.
- Colour classes in the analysis and in the CSV: area, % of the fish, number of spots and single spot areas per
  class, reflections excluded and reported; merge distance and minimum spot size; check image per fish.
- Comparison with areas measured by hand: mean difference, mean absolute error, correlation, Bland-Altman plot.
- Models stored as plain arrays (no pickle). scikit-learn added to the locked dependencies; installer uses
  binary wheels only.

## 0.1.3 (2026-10-22)
Security review fixes:
- Locked, hash-checked dependencies (`requirements-win.txt`), non-editable install, no pip upgrade.
- Model weights downloaded by the app and checked against SHA-256 before loading; weights-only loading.
- Ultralytics telemetry, runtime installs and network checks turned off.
- OCR model download hash-checked; "Open the original" opens image files only; a new project never
  overwrites an existing one; CSV text cannot become an Excel formula.
- Installer: no `-ExecutionPolicy Bypass`, working folder in `%LOCALAPPDATA%`, Python 3.12 required.

## 0.1.2 (2026-10-22)
- Fish id read from the label in the photo (QR or OCR) with list of valid ids from Excel/CSV; id editable in Review.
- "Print labels with QR code" button.

## 0.1.1 (2026-10-22)
- Tested on 18 real photos: recursive photo folders (sub-folder = session), quality warnings from the outline
  shape, cropped check images, one click on the fish in Review to redo the outline.

## 0.1.0 (2026-10-22)
- First version: projects with photo folders, colour correction with the ColorChecker (nominal, reference
  photo, single calibration, ImageJ CSV), automatic fish outline (YOLOE + SAM 2.1), scale from graph paper,
  length/area/depth, file-name fields and EXIF, Review screen, CSV for R, Windows installer.

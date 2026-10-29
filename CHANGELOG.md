# Changelog

## Planned
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

## Unreleased
- New `training_kit/` (separate from the app, not in its release zip): standalone pre-training on a PC with an
  NVIDIA card, without Guppy Colour or Claude installed. It places the van der Bijl et al. 2025 annotations (FRDR,
  CC BY 4.0) on their original photos and trains an outline model (body + caudal fin) and a landmark model (snout,
  eye, end of the peduncle, tail tip, so SL and TL), checks both on held-out fish against their annotations
  (`report.html`) and writes model files in the app's export format. See `training_kit/README.md` and
  `MODELS_AND_TRAINING.md`. Kit 0.2.0 adds a colour model (orange, black, body without colour) from the van der
  Bijl masks, `colours_vanderbijl.zip`, to import in Training and refine with Ritocco; kit 0.2.1 reinstalls its
  packages when the kit folder is new and finds its data after the folder is moved. Kit 0.2.2 weights each colour
  class by its true share of the fish (0.2.1 over-predicted orange and black by about 3 points); kit 0.3.0 records
  that share in the model (`class_share`, used by app 0.13.0 Ritocco) and teaches the landmark model the top and
  bottom of the caudal fin base. Use the latest kit.

## 0.13.0 (2026-10-29)
After installing, **run the Analysis again**: every photo is analysed once more because SL, TL and the colours are
measured in a new way (outlines redone by hand, clicked landmarks, typed ids and Review decisions are kept).
- **SL and TL follow the fish**: SL goes from the snout to the end of the caudal peduncle landmark along the midline
  of the outline, so a bent fish is no longer measured short; TL = SL + the tail fin (end of the peduncle to the tail
  tip, along the midline). Without the tail tip landmark TL is still the outline length when the fin is all in it.
  `caudal_length_mm` is also along the midline.
- **Rounded end of the peduncle**: two new optional landmarks, *top* and *bottom of the caudal fin base*
  (`caudal_base_top`, `caudal_base_bottom`, clicked before and after the end of the peduncle). With them the body
  ends on the arc through the three points, which gives `body_area_mm2` and `caudal_area_mm2`. SL still ends at
  the peduncle landmark, not at the middle of the two new points. Landmark models without the new points still
  work (the points are just not predicted); click them on some photos and train with Ritocco to learn them.
- **Colours start behind the head**: 2 mm behind the eye landmark; the head counts as body without colour, so the
  eye is never black and `pct_<class>` stays a share of the whole fish. Without an eye: 3.5 mm behind the snout,
  flag `colour_start_from_snout`; without either: whole fish, flag `colour_start_missing`. Set in the profile
  (`colour_start`); drawn as a yellow dotted line in Results. Moving a landmark re-measures the colours.
- **Landmarks guide the outline**: a landmark outside the outline (typically the tail tip on a transparent caudal
  fin) is used as a click to add the missing part, as *Add to the outline* does in Review. Flag
  `outline_grown_to_landmarks`.
- **Review**: Accept / Exclude / size only / size + colour are now in a box *The whole photo (all views)* and act
  on the whole photo; the photo stays on screen after every fix (also after a scale) until you decide. The Label
  view shows what was read (id, group, date, source, other readings, all the text) and the id box offers the other
  readings to pick from.
- **Size check against the folder**: Settings > Scale > *Check* (default 10 %) sends to Review the photos whose
  px/mm or fish length is further than that from the median of their folder (flags `scale_off_session`,
  `size_off_session`). It changes no number and needs no new analysis.
- **Colour Ritocco keeps the true share of each class**: the new model no longer paints more orange or black than
  there is (with the van der Bijl colour model, about 3 points too many). Models record the share in `class_share`.
- Settings > Fish ID: the formats are explained (`\d`, `{3}`, `/`) and **Try the formats on** checks a text.
- Training says when Settings measures size only, with a button to measure colours again.

## 0.12.0 (2026-10-27)
- **Optional colour classes**: the iridescent class is now optional (`"optional": true` in the profile). Settings >
  What to measure > *Measure the colour class Iridescent* switches it on or off. New projects start with it off;
  projects made before keep it on (switch it off there by hand if you wish: the photos are then marked for a new
  analysis). When off, its pixels count as body without colour, Training has no brush for it, and its CSV columns
  (`area_iridescent_mm2`, `pct_iridescent`, `n_spots_iridescent`, `spots_iridescent_mm2`) are `NA` (not 0).
- **Ritocco for colour models**: Training > Start from: a colour model that keeps its training pixels (e.g. one
  pre-trained on the van der Bijl 2025 orange and black masks, imported with Import model…) is used as a start, and
  your brush strokes count 3 times more than its pixels. Every colour model trained from now on keeps up to 10,000
  of its training pixels per class (a few MB), so it can be a start in another project. Models trained before 0.12
  cannot be a start (train them again).
- Import model (colours) accepts a model with only some of the project's classes (e.g. orange, black and body
  without colour): the missing classes are not measured by it until you paint them and train with Ritocco.
- New guide `MODELS_AND_TRAINING.md`: which models exist, where they are saved, which file to import on which
  screen, where the trained models are, and how to train on each PC (app and GuppyTraining kit).

## 0.11.0 (2026-10-26)
- **Outline models**: Settings > Fish outline can use an imported outline model instead of the automatic detector,
  e.g. `outline_vanderbijl.zip` from the training kit (body and tail fin drawn separately). The body area and
  whether the tail fin is in the outline then come from the model. Where it finds no fish, or its outline is in
  pieces, cut by the border or oddly shaped, the automatic detector is used for that photo (new flags
  `outline_model_missed`, `outline_model_rejected`, `outline_model_error`). New CSV column `outline_source`.
  Changing the outline model marks the photos for a new analysis; projects that do not use one stay as they are.
- **Ritocco**: Landmarks > Start from: training can start from a landmark model of the project (e.g. the imported
  `landmarks_vanderbijl.zip`) and refine it with your clicks, in at most 60 rounds with gentle steps.
- Landmark models can say which landmarks they were never trained on (the training kit's has no dorsal fin base
  and gonopodium base): those are no longer predicted. A model's own image size is used for predictions.
- Export / Import model works for outline models too.

## 0.10.0 (2026-10-23)
- **Landmarks win over the outline** (new module `sizes.py`), as soon as they are clicked on screen 8 or predicted:
  - TL = snout -> tail tip whenever the tail tip is marked (`total_length_source = landmarks`), even if the
    outline has the fin;
  - the end of the body is the line across the fish at the peduncle landmark: `body_area_mm2` is the outline
    before it (`body_area_source = landmarks`), `caudal_area_mm2` the outline after it (only when the fin is in
    the outline);
  - the fin counts as in the outline when the outline reaches 85% of the way from the peduncle to the tail tip;
  - new `caudal_length_mm` = peduncle end -> tail tip;
  - **maximum depth** across the body (perpendicular to snout -> peduncle) at the base of the gonopodium
    (`depth_source = landmarks`), else the widest point of the outline.
  Removing the landmarks gives the outline values back. Results draws these lines in blue (TL dashed).
- **Review** shows each photo in separate views, **Fish** (outline and measures), **Label** (the label close up, or
  the whole photo when it was not found, with the id box) and **Scale** (the whole photo, Confirm / Calibrate).
  Views with something to check have a warning sign and list their problems; the screen opens on the most
  important one.
- **Add to the outline** is now a click mode in the Fish view ("A click on the fish: redoes the whole outline /
  adds the part you click"). After a click the photo stays on screen with the new outline, and a message says how
  much the outline grew, whether the tail fin is now in it and what is left to check; if nothing new was found, it
  says so.

## 0.9.0 (2026-10-23)
- Results: the table now has `standard_length_mm` and `total_length_mm`; a line says on how many photos SL is
  known and how to get it (landmarks on screen 8). Click a row to see the photo with its measures drawn
  (outline, midline = mask length, snout, end of the outline, caudal peduncle, maximum depth, 5 mm bar) and the
  landmarks with the SL line, plus a colour legend. Review shows the same drawing.
- Fixed: on photos with the fish upright (head up or down) the "head on the left/right" setting put the snout on
  the tail, so the body area and the maximum depth were measured from the wrong end. The setting now applies only
  to fish lying sideways; upright fish are judged by the eye. Run the Analysis again on such photos.
- Training and Landmarks screens: the crop around the fish leaves more room at the head and the tail (30% of the
  fish length), so the transparent caudal fin is no longer cut off.

## 0.8.2 (2026-10-23)
- Fixed: landmark training stopped with "'NoneType' object has no attribute 'write'" when the app was opened from
  the desktop icon (no console window, so the training progress text had nowhere to go). That text now goes to
  `%LOCALAPPDATA%\GuppyColour\guppycolour.log`, rewritten at every start.

## 0.8.1 (2026-10-23)
- Every screen scrolls when the window is too small for it, instead of cutting the bottom or squashing the
  texts on top of each other.

## 0.8.0 (2026-10-23)
- Review: **Add to the outline**: the next click joins the part clicked (e.g. a transparent caudal fin) to the
  outline instead of redoing it (SAM prompted with the click and a point inside the fish; only pieces touching
  the outline are kept).
- Total length from the landmarks (snout to tail tip) when the caudal fin is not all in the outline; new column
  `total_length_source` (`outline`, `landmarks`).
- Caudal fin detection: the peduncle is the narrowest point before the widest part of the fin; a round fin's own
  tapering edge was taken for it, so open fins were reported as `tail_incomplete` (4 of 11 test photos fixed).
- Installer says how big and how long the package step is, so a long "Installing collected packages" does
  not look stuck.

## 0.7.1 (2026-10-23)
- Landmark training failed on Windows ("Download failure … Arial.ttf, environment may be offline"): ultralytics
  does not see Windows' `arial.ttf` and tries to download one. matplotlib's DejaVuSans is now put in its place
  in the ultralytics settings folder, so nothing is downloaded.

## 0.7.0 (2026-10-23)
- Training and Landmarks: progress bar while training (photos read, then trees grown; training rounds for
  landmarks).
- **Export model… / Import model…**: a trained colour or landmark model as one `.zip` (manifest with the SHA-256
  of each part) to reuse in another project or share. Import refuses damaged files, unexpected contents, and
  models whose colour classes or landmarks differ from the project's profile; it is saved as the next version with
  `imported_from`.
- Colour forest grown 10 trees at a time (same settings) to report progress.

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

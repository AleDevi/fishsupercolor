# Guppy Colour

Batch measurement of fish colour patterns from photos. One fish per photo: the app corrects the colour with
the colour chart, finds the fish outline, reads the scale from graph paper, measures length and area, reads
the fish id, measures the area of each colour class with a model you train in the app, and writes one CSV
row per fish, ready for R.

**Scope.** The app is tuned first for male guppies (*Poecilia reticulata*), but it is meant for any fish, both
sexes, and later possibly other animals. Nothing about guppies is hard-wired: body parts, colour classes and
landmarks (with the ones used for SL, TL and body depth) live in a profile per species and sex
(`guppycolor/profiles/`), and new features are built on that. Profiles for female guppies and other species
are planned; photos of them can already be analysed for outline, size and colour.

- [Install (Windows)](#install-windows)
- [Workflow](#workflow)
- [The CSV](#the-csv)
- [Warnings (flags)](#warnings-flags)
- [Project folder](#project-folder)
- [How it works](#how-it-works)
- [Security](#security)
- [Command line](#command-line)
- [For developers](#for-developers)
- [What is missing](#what-is-missing)

## Install (Windows)

You need Python 3.12 from python.org (no other Python knowledge needed). Python 3.13 does not work yet: `rapidocr-onnxruntime` has no 3.13 release. Both versions can be installed side by side; the installer picks 3.12.

1. Unzip `GuppyColour_<version>.zip`, for example into `Documents`.
2. Double-click `install_windows.bat`. It takes a few minutes and needs internet.
3. The **Guppy Colour** icon appears on the desktop. The unzipped folder is not needed after this.

The first analysis downloads the fish-finding models (about 830 MB, once) into
`C:\Users\<you>\.guppycolor\models` and checks them. If the icon does not open, run `start_guppycolour.bat`:
it shows the error messages. When opened from the icon, the technical messages (for example the progress of
landmark training) are written to `%LOCALAPPDATA%\GuppyColour\guppycolour.log`, rewritten at every start: send
that file along when something goes wrong.

To update, unzip the new version and run `install_windows.bat` again. Projects are kept. The update reuses the
installed packages: only the app itself and any package whose version changed are downloaded.

**Disk space.** What the app keeps: `%LOCALAPPDATA%\GuppyColour\venv` (Python packages, about 1.5 GB) and
`C:\Users\<you>\.guppycolor\models` (models, about 830 MB, downloaded once). Safe to delete: old
`GuppyColour_*.zip` files and their unzipped folders, and the pip cache left by versions before 0.6.1
(`py -3.12 -m pip cache purge`, or delete `%LOCALAPPDATA%\pip\Cache`). The installer no longer fills that cache.

## Workflow

The window has one screen per step, on the left.

### 1 Project
A project is one experiment: a folder that holds the settings, the results and the exports. **New project**
asks for an empty folder and a name, then for the first photo folder. **Add folder** adds more photo folders
at any time; sub-folders are included, and each folder is a *session* (column `folder` in the CSV). Only
new or changed photos are analysed. Photos are never modified.

### 2 Settings
Changes are saved immediately. Changing a setting that affects the numbers marks the photos for a new
analysis at the next Start.

- **Colour correction**
  - *None*.
  - *Nominal values*: the ColorChecker 24 found in each photo is mapped to its nominal values (3D thin-plate spline).
  - *Reference photo*: every chart is mapped to the chart of one chosen photo. The reference can also be
    imported as a CSV from the ImageJ plugin (`Patch,Name,R,G,B`, 0–255).
  - *Single calibration*: the transform measured on the reference photo is applied to every photo, also to photos without a chart.
  Photos without a chart are measured uncorrected and flagged `no_chart`.
- **Scale**: automatic from the graph paper (smallest square = 1 mm, measured over at least 20 squares = 2 cm),
  by hand (a known length drawn on the photo in Review), a fixed px/mm value (**Measure on a photo…** draws it on
  one photo), or none (pixels). A doubtful automatic scale goes to Review and, until it is confirmed, the CSV uses
  the median of the session.
- **Body parts**: caudal, dorsal fin and gonopodium can be included or excluded. Recorded in the CSV; the
  separation itself comes with the body-part model.
- **Head of the fish**: left (default), right, or guessed in each photo. Used for lengths and alignment.
- **File names**: a pattern such as `{id}_{tank}_{date}` turns parts of the file name into CSV columns.
- **Fish ID**: where the fish id comes from:
  - *No id*.
  - *Label in the photo*: QR code first, else the written text (OCR), checked against the formats
    (`M\d{3}`, `G\d{1,2}`, `\d{1,2}/\d{2}`, editable) and, if given, against a list of valid ids (Excel/CSV).
  - *File name*: the `{id}` part of the pattern.
  - *Data table + sequence*: a sheet with the fish id and the name of its photo (e.g. `SL_pictureID`).
    Photos not in the table take the id of the table photo before or after them in shooting order,
    checked against the label and image similarity; disagreements go to Review.
  With *No id* or *File name* you can still read the label for its date and group.
- **What to measure**: size and colour on every fish, or size only; a pattern on the fish id or file name
  can mark size-only fish (e.g. `_F\d+$` for females such as `F100_F1`). Folders can be set to size only in
  **Project**, single photos in **Review**.
  **Print labels with QR code** makes an A4 PDF of labels (QR + large text) from a fish list.

### 3 Analysis
**Start analysis** runs on all pending photos; it can be stopped and resumed. Thumbnails show the outline
found on each fish.

### 4 Review
Photos with errors or warnings, most uncertain first. **Accept** (Enter) or **Exclude**. Each photo has three views;
the ones with a warning sign have something to check, listed in red above the picture, and the screen opens on
the most important one:

- **Fish**: the outline with its measures drawn (legend in 5 Results). Choose what a click on the fish does:
  *redoes the whole outline* (fish not found, or wrong outline) or *adds the part you click to the outline*
  (transparent fins, often the caudal fin: click on the missing part, close to the fish; repeat for more parts).
  After a click the photo stays on screen with the new outline, and a message says how much the outline grew,
  whether the tail fin is now in it and what is left to check.
- **Label**: the label close up (the whole photo when it was not found) with the text read; type the right id and
  **Save id** (it wins over the reading from then on).
- **Scale**: the whole photo. When the automatic scale is doubtful a red line should cover 2 cm (20 squares).
  **Confirm scale** if it does; otherwise **Calibrate manually**: zoom in (wheel), click the two ends of a known
  length (2 cm by default, the longer the more precise), type its length in mm, and tick *Use it also for the
  other photos of the folder* when the camera did not move. The scale drawn by hand stays for later analyses.

### 5 Results
Table of the main measures (SL, TL, mask length, depth, areas) and **Export CSV for R** (`export/measurements.csv`).
Above the table: on how many photos the standard length (SL) is known. SL needs the landmarks (snout and end of the
caudal peduncle), clicked on screen 8 or predicted by a landmark model; without them it is `NA`.

Click a row to see how that fish was measured. The lines are drawn on the photo at analysis time:
- magenta: outline of the fish (fish and body areas);
- cyan: midline, from the snout (green) to the end of the outline (red) = `mask_length_mm`;
- orange: end of the body at the caudal peduncle, only when the tail fin is in the outline (body area stops there);
- yellow: maximum depth; white bar: 5 mm, when the scale is known;
- blue: the landmarks and what comes from them, as they are now (they change as soon as you click them on screen 8):
  SL line, TL (dashed, snout to tail tip), the end of the body across the peduncle, the maximum depth across the
  body at the gonopodium base.

**Landmarks win over the outline.** When the landmarks are there (clicked or predicted): TL = snout to tail tip;
the body ends at the peduncle landmark (body area before it, caudal fin area after it, when the fin is in the
outline); the maximum depth is measured across the body at the base of the gonopodium. The `*_source` columns of
the CSV say where each value came from.

Photos analysed before 0.9.0 show only the outline: run the Analysis again to get the drawing.

### 6 Pattern and clusters
After the analysis with a colour model, open screen 6 and press **Align and compare**. Every fish is stretched onto the average body shape, so a spot in the same place on two fish lands on the same pixels. Coordinates are in body lengths: x = 0 at the snout and 1 at the caudal peduncle, y from the snout–peduncle line, positive towards the back.

Choose how to compare fish:
- **Colour maps**: how much of each colour overlaps pixel by pixel (Jaccard). *Tolerance* forgives small misalignments.
- **Spots as points and lines**: round spots are points, elongated ones are lines; spots are paired one to one, and two spots more than 0.15 body lengths apart do not pair.
- **Colour amounts only**: how much of each colour, wherever it is.

Move the **similarity** slider (or click on the tree): fish more similar than that value end up in the same cluster. Each cluster shows its mean colour map (stronger colour = more fish have it there); click a cluster to see its fish with their spots. If the labels have a group, the screen shows how well the clusters match the groups (adjusted Rand: 1 = same groups, 0 = chance), which is the way to check the method on fish of known groups.

**Export CSV for R** writes the files in `export/pattern/`. In R: `read.csv("pattern_fish.csv")`; for the distances `as.dist(as.matrix(read.csv("pattern_distances.csv", row.names = 1)))`.

Landmarks clicked in screen 8 (or placed by the landmark model) are used: the snout fixes the head, and the named landmarks marked for Procrustes in the profile join the alignment when every fish has them. Without them, the end of the body is the narrowest point of the outline (the caudal peduncle guess); the dorsal side is taken as the top of the photo once the head is put on the left.

### 7 Training (colour classes)
1. Type your name as **Annotator** (stored with the model).
2. On a few fish (5–10 to start), paint a few strokes of each class: one brush per colour class of the
   profile (keys 1, 2, 3…), plus *Body without colour* and *Light reflection*; **E** is the eraser. Mouse
   wheel zooms, right button drags. Strokes are saved as you paint.
3. **Train**. The model (`models/colours_vN`) is used in the analysis straight away and **Show what the
   model sees** paints its prediction over the photo. Add strokes where it is wrong and train again; each
   training makes a new version, and **Use in the analysis** picks one.
4. Run the **Analysis** again: the CSV gets the colour columns (below) and `check/*_colours.jpg` shows the
   classes painted on each fish.
5. **Most uncertain first** sorts the photos by the model's confidence: strokes there help most.
6. **Merge spots closer than** (mm) counts nearby spots as one, as when they are measured together by hand;
   **Ignore spots smaller than** (mm²) drops specks.
7. **Compare with measures made by hand**: a table (xlsx or csv) with the fish id and the area of each class
   in mm², and optionally total length and fish area (columns are recognised by name, e.g. `OrangeT`,
   `BlackT`, `Totallength`, `Area`). It also works without a colour model, for sizes only. The app matches fish by label id, file-name id or file name, and reports per class: number of
   fish, mean difference, mean absolute error, correlation and Bland-Altman limits of agreement, with a plot
   (`export/validation_<model>.png`) and the pairs (`.csv`). Spots cannot be matched one by one, so the
   comparison is on the totals.

**Where the models are.** In the project folder, `models\colours_v1.npz` + `colours_v1.json` (date, annotator,
photos, pixels per class); every training adds v2, v3… A bar shows the training progress.

**Export model… / Import model…** save the model chosen in *Use in the analysis* as one `.zip` file, to use it in
another project or give it to a colleague. Import checks the file (fixed contents, SHA-256 of each part) and that
its colour classes are those of the project's species profile, then adds it as the next version and puts it in use;
the model keeps where it came from (`imported_from` in its `.json`).

Strokes made in Windows Paint can be imported: save a copy of the photo as `<photo name>_strokes.png` next
to it and paint with Paint's orange (carotenoid), green (black), blue (iridescent), yellow (body without
colour) and purple (reflection), then **Import strokes from Paint**.

### 8 Landmarks
1. The list shows the analysed fish (photos with clicks first, marked ✓). The profile's landmarks are on the
   right in order: for the guppy male *snout, eye, dorsal fin base, gonopodium base, end of the caudal
   peduncle, tip of the tail*.
2. Click each landmark on the fish; the highlighted one is next. **S** skips an optional landmark,
   **Backspace** undoes, **N** goes to the next photo. Mouse wheel zooms, right button drags. Clicks are saved
   at once and the **standard length** (snout to end of the caudal peduncle) goes into the results straight
   away: no new analysis needed.
3. With 10–30 photos clicked, **Train** makes a landmark model (YOLO-pose, `models/landmarks_vN`). It starts
   from the small public `yolo11n-pose.pt` weights (downloaded once and checked against their SHA-256). A GPU
   takes minutes, a CPU much longer. The new model is put in use; run the **Analysis** again to get SL (and
   the landmarks) on every fish. Landmarks the model is unsure of (confidence < 0.5) are left out.
   A bar shows the training rounds; it may stop before the last one when the model no longer improves.
   **Export model… / Import model…** work as in Training (the landmarks must be the same as in the project's
   profile). Landmark models contain neural-network weights: import only files from people you trust.
4. Clicked landmarks always win over the model. The CSV column `landmarks_source` says which one was used
   (`manual`, `model`, `NA`).

## The CSV

`export/measurements.csv`: comma separated, decimal point, `NA` for missing values, `TRUE`/`FALSE`. In R:
`d <- read.csv("…/export/measurements.csv")`.

| Column | Meaning |
| --- | --- |
| `file`, `folder` | photo file name and its session folder |
| fields of the file-name pattern | e.g. `id`, `tank`, `date` |
| `date_taken`, `camera`, `exposure_s`, `iso`, `focal_mm` | from the photo's EXIF |
| `colour_chart` | a colour chart was found and used |
| `correction` | `none`, `nominal`, `reference`, `shared` |
| `reference_photo` | reference photo of the colour correction |
| `chart_error_de00` | leave-one-out error of the chart correction (CIEDE2000) |
| `px_per_mm`, `scale_source` | scale and where it came from (`graph_paper`, `line` = drawn on this photo, `line_folder` = drawn on another photo of the folder, `session_median`, `fixed`, `none`, `not_found`, `to_set`) |
| `parts_included` | body parts included, e.g. `body+caudal+dorsal` |
| `standard_length_mm` | SL, snout to end of the caudal peduncle, from the landmarks (`NA` without them) |
| `landmarks_source` | `manual` (clicked), `model` (landmark model) or `NA` |
| `total_length_mm` | TL: the outline length when the whole caudal fin is in it; otherwise snout to tail tip from the landmarks; else `NA` (flag `tail_incomplete`) |
| `total_length_source` | `outline`, `landmarks` or `NA` |
| `mask_length_mm` | length of the outline along the fish |
| `caudal_length_mm` | end of the peduncle to tail tip (landmarks) |
| `max_depth_mm`, `depth_source` | maximum body depth: across the body at the gonopodium base (`landmarks`) or widest point of the outline (`outline`) |
| `fish_area_mm2`, `body_area_mm2`, `body_area_source` | area of the outline; area without the caudal fin, cut at the peduncle landmark (`landmarks`) or where the outline narrows (`outline`) |
| `caudal_area_mm2` | outline after the peduncle landmark, only when the tail fin is in the outline |
| `area_<class>_mm2` | area of each colour class (with a colour model) |
| `pct_<class>` | % of the fish area (reflections left out) |
| `n_spots_<class>` | number of separate spots |
| `spots_<class>_mm2` | area of each spot, largest first, separated by `;` |
| `reflection_pct` | % of the fish covered by light reflections (left out of every area) |
| `colour_model`, `colour_confidence` | model used and its mean confidence on the fish |
| `label_id`, `label_group`, `label_date`, `label_text` | fish id, group and date, raw text read |
| `label_source` | `qr`, `ocr`, `filename`, `table`, `sequence`, `manual` |
| `label_confidence`, `label_needs_review` | confidence of the reading; whether to check it |
| `analysis` | `size_colour` or `size` (size only: colour columns NA) |
| `fish_brightness` | median of max(R,G,B) on the fish (0–255) |
| `saturated_pct` | % of fish pixels burnt out (max(R,G,B) ≥ 250) |
| `sharpness` | variance of the Laplacian on the fish (compare within a session) |
| `fish_confidence` | confidence of the fish detector |
| `status` | `ok`, `to_check`, `error`, `reviewed`, `excluded` |
| `flags` | warnings, separated by `;` (below) |
| `path` | full path of the photo |

In R, the spots of one class: `strsplit(d$spots_orange_mm2, ";")`.

## Warnings (flags)

| Flag | Meaning | Sends to Review |
| --- | --- | --- |
| `no_chart` | no colour chart: photo not corrected | |
| `reference_missing` | colour reference not chosen | yes |
| `scale_doubtful`, `scale_not_found` | graph paper not read reliably | yes |
| `scale_from_median` | scale taken from the session median | |
| `scale_to_set` | scale by hand chosen and not drawn yet on this photo or its folder | yes |
| `fish_not_found` | no fish found (click on it in Review) | error |
| `fish_at_border` | the fish touches the photo border | yes |
| `mask_in_pieces` | the outline is broken into pieces | yes |
| `unusual_shape` | length/depth ratio outside 3–10 | yes |
| `area_outlier` | area outside 0.5–2× the session median | yes |
| `tail_incomplete` | caudal fin not all in the outline: no TL | |
| `label_uncertain` | label read with low confidence | yes |
| `date_mismatch` | date on the label differs from the EXIF date | |
| `colours_uncertain` | colour model unsure on this fish (a good one to paint) | |
| `overexposed` | more than 1% of the fish has max(R,G,B) ≥ 250 | |
| `blurry` | sharpness below 40% of the session median | |
| `sl_landmarks_missing` | landmarks present but snout or end of the peduncle missing: no SL | |
| `landmark_model_error` | the landmark model file changed or is missing: not used | |

## Project folder

```
<project>/
  project.json            settings and photo folders
  colour_reference.csv    colour reference (Patch,Name,R,G,B)
  results/<photo>.json    everything measured on each photo (pixels)
  masks/<photo>.png       fish outline; <photo>_manual.png = outline redone in Review
  check/<photo>.jpg       outline and measures drawn on the fish; <photo>_colours.jpg = colour classes
  training/strokes/       painted strokes (one PNG per photo, value = brush number)
  training/images/        colour-corrected photos used for training (cache)
  models/colours_vN.npz   colour models (+ .json with date, annotator, photos, pixels per class)
  scale/<photo>.json      line drawn by hand (ends in photo pixels, mm); scale/folders.json = px/mm per folder
  landmarks/<photo>.json  landmarks clicked by hand (photo pixels, annotator, date)
  models/landmarks_vN/    landmark models (+ landmarks_vN.json with landmarks, photos, SHA-256 of the weights)
  training/landmarks_dataset/  crops and labels made for the last landmark training
  export/                 measurements.csv, validation_*.csv/png
```

The whole folder can be copied to another computer (e.g. one with a GPU) and opened there.

## How it works

- **Fish outline**: YOLOE with the text prompt "fish" finds the fish, SAM 2.1 draws its outline
  (no training needed). One click in Review gives SAM a point instead.
- **Colour**: ColorChecker 24 located automatically; 3D thin-plate spline from measured to target values
  (Menesatti et al. 2012).
- **Scale**: period of the graph-paper grid by autocorrelation around the fish, fitted over the whole paper
  patch (at least 20 squares, else doubtful); a 2 cm line along the grid is kept to show it in Review.
- **Measures**: length along the outline's midline, maximum depth, areas; the eye marks the head.
- **Colour classes**: per-pixel features (CIELAB colour smoothed at 3 scales, edges, blob filters, local
  contrast) and a random forest trained on your strokes (as in ilastik). Pixels are classified inside the
  fish outline; spots are connected regions of a class.
- **Landmarks**: a YOLO-pose model fine-tuned on your clicks, run on a crop around the fish outline
  (no left-right flips in training, so left and right landmarks stay distinct).
- **Labels**: QR code, else OCR (PP-OCR via onnxruntime) constrained to the expected formats.

## Security

- Python packages come from `requirements-win.txt`: fixed versions, each with its SHA-256 hash, binary wheels
  only. A different file stops the installation.
- The model files are checked against their SHA-256 before being opened; ultralytics loads them with
  `weights_only` and never installs packages or sends usage statistics.
- Landmark models trained in the app are checked against the SHA-256 written when they were trained before
  being loaded (weights only).
- Colour models are plain arrays (`.npz`, loaded without pickle) plus JSON. Profile and model names read from
  a project are restricted to letters, digits, `_` and `-`.
- No administrator rights, nothing outside `%LOCALAPPDATA%\GuppyColour`, `~\.guppycolor` and the project folder
  is written. Photos are only read.
- Text that Excel would treat as a formula is prefixed with `'` in the CSV.

## Command line

```
guppycolor                       open the window
guppycolor new PROJECT --name X FOLDER...
guppycolor run PROJECT           analyse pending photos and write export/measurements.csv
```

## For developers

- `guppycolor/engine/`: stateless algorithms (`colour`, `segment`, `scale`, `morpho`, `labels`, `pixels`).
- `guppycolor/project.py`, `pipeline.py`, `export.py`: project, analysis of one photo, CSV.
- `guppycolor/training.py`, `verify.py`: colour-class training and comparison with manual measures.
- `guppycolor/modelshare.py`: export/import of trained models as `.zip` (manifest with SHA-256).
- `guppycolor/landmarks.py`: clicked landmarks, standard length, YOLO-pose dataset, training and prediction.
- `guppycolor/pattern.py`, `engine/pattern.py`: alignment and clustering of the colour patterns.
- `guppycolor/gui/`: PySide6 window (`main.py`, `training_page.py`, `pattern_page.py`, `landmarks_page.py`).
- `guppycolor/profiles/*.json`: species profiles.
- Tests: `python -m pytest tests` (synthetic photos, no model download).
- GitHub Actions: `tests` runs the tests on Windows with Python 3.12 and the locked packages at every push and
  pull request; `release` publishes `GuppyColour_<version>.zip` and its `.sha256` when a tag `v<version>` is
  pushed (the tag must match `guppycolor/__init__.py`). Tag a release only after its security re-check.
- Dependencies: edit `requirements.in`, then
  `uv pip compile requirements.in --python-platform windows --python-version 3.12 --generate-hashes -o requirements-win.txt`.

## What is missing

- Separate body parts (caudal, dorsal, gonopodium): annotation screen and model next.
- Spectra estimated from the photos (module ready, not yet in the app).
- Update check from GitHub releases (off while the repository is private).
- RAW photos (now JPEG/PNG/TIFF): linear demosaicing before the colour correction.

## License

GNU Affero General Public License v3.0 or later (see `LICENSE`), as required by ultralytics.

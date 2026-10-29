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

Models (which ones, where they are saved, how to load them, how to train them on each PC, the GuppyTraining
kit): [MODELS_AND_TRAINING.md](MODELS_AND_TRAINING.md).

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
- **Fish outline**: *Automatic* (YOLOE + SAM 2.1, no training, works on any fish) or an imported **outline
  model** (**Import outline model…**), e.g. `outline_vanderbijl.zip` from the training kit, pre-trained on the van
  der Bijl 2025 guppy photos. An outline model draws the body and the tail fin separately: the body area and
  whether the fin is in the outline then come from it. Where it finds no fish, or its outline is in pieces, cut by
  the border or oddly shaped, the automatic detector is used for that photo (flags `outline_model_missed`,
  `outline_model_rejected`). The CSV column `outline_source` says which drew each outline. On photos unlike the
  ones it was trained on (graph paper with closed fins, dark or tight framing) the automatic one can be better:
  try both on a few photos.
- **Scale**: automatic from the graph paper (smallest square = 1 mm, measured over at least 20 squares = 2 cm),
  by hand (a known length drawn on the photo in Review), a fixed px/mm value (**Measure on a photo…** draws it on
  one photo), or none (pixels). A doubtful automatic scale goes to Review and, until it is confirmed, the CSV uses
  the median of the session. **Check** (default 10 %) sends to Review the photos whose scale or fish length is
  further than that from the median of their folder (at least 3 photos; 0 switches it off; raise it when a folder
  holds fish of very different sizes). It does not change the numbers.
- **Body parts**: caudal, dorsal fin and gonopodium can be included or excluded. Recorded in the CSV; the
  separation itself comes with the body-part model.
- **Head of the fish**: left (default), right, or guessed in each photo. Used for lengths and alignment.
- **File names**: a pattern such as `{id}_{tank}_{date}` turns parts of the file name into CSV columns.
- **Fish ID**: where the fish id comes from:
  - *No id*.
  - *Label in the photo*: QR code first, else the written text (OCR), checked against the formats
    (`M\d{3}`, `G\d{1,2}`, `\d{1,2}/\d{2}`, editable) and, if given, against a list of valid ids (Excel/CSV).
    How to read a format: a letter stands for itself (`M`), `\d` is any digit 0–9, `{3}` means exactly 3 of what
    comes before it and `{1,2}` 1 or 2, `/` is the slash written on the label. So `M\d{3}` reads M001…M999 and
    `\d{1,2}/\d{2}` reads dates such as 7/03 or 12/10. **Try the formats on** checks a text you type against them.
  - *File name*: the `{id}` part of the pattern.
  - *Data table + sequence*: a sheet with the fish id and the name of its photo (e.g. `SL_pictureID`).
    Photos not in the table take the id of the table photo before or after them in shooting order,
    checked against the label and image similarity; disagreements go to Review.
  With *No id* or *File name* you can still read the label for its date and group.
- **What to measure**: size and colour on every fish, or size only; a pattern on the fish id or file name
  can mark size-only fish (e.g. `_F\d+$` for females such as `F100_F1`). Folders can be set to size only in
  **Project**, single photos in **Review**. *Measure the colour class …* switches each optional colour class of the
  profile on or off (guppy male: iridescent, off in new projects).
  **Print labels with QR code** makes an A4 PDF of labels (QR + large text) from a fish list.

### 3 Analysis
**Start analysis** runs on all pending photos; it can be stopped and resumed. Thumbnails show the outline
found on each fish.

### 4 Review
Photos with errors or warnings, most uncertain first. Each photo has three views; the ones with a warning sign have
something to check, listed in red above the picture, and the screen opens on the most important one. Fix what each
view needs (the photo stays on screen after every fix), then decide on the whole photo in the box **The whole photo
(all views)**: **Accept the photo** (Enter), **Exclude the photo**, **Measure size only** or **Measure size +
colour**. Only these move to the next photo.

- **Fish**: the outline with its measures drawn (legend in 5 Results). Choose what a click on the fish does:
  *redoes the whole outline* (fish not found, or wrong outline) or *adds the part you click to the outline*
  (transparent fins, often the caudal fin: click on the missing part, close to the fish; repeat for more parts).
  After a click the photo stays on screen with the new outline, and a message says how much the outline grew,
  whether the tail fin is now in it and what is left to check.
- **Label**: the label close up (the whole photo when it was not found) and what was read: the fish id, group and
  date found, where they came from (QR, OCR, file name, table), the other possible ids and all the text read. The
  id box lists the other readings: pick one or type the right id, then **Save id** (it wins over the reading from
  then on).
- **Scale**: the whole photo. When the automatic scale is doubtful a red line should cover 2 cm (20 squares).
  **Confirm scale** if it does; otherwise **Calibrate manually**: zoom in (wheel), click the two ends of a known
  length (2 cm by default, the longer the more precise), type its length in mm, and tick *Use it also for the
  other photos of the folder* when the camera did not move. The scale drawn by hand stays for later analyses.
  The Scale view also opens when the scale or the fish length is far from the rest of the folder (Settings >
  Scale > Check): check the line, or the fish.

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
3. **Start from** (optional, *Ritocco*): a colour model that keeps its training pixels, e.g. one pre-trained on the
   van der Bijl 2025 orange and black masks and imported with **Import model…**. Your strokes count 3 times more
   than its pixels, so a few strokes adapt it to your light and camera. Each class keeps its true share of the fish
   (from the start model, when it records it), so the new model does not paint more orange or black than there is. *nothing: only my strokes* trains from your
   strokes alone. **Train**. The model (`models/colours_vN`) is used in the analysis straight away and **Show what the
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

**Colours start behind the head.** Colours are measured from a line across the fish 2 mm behind the eye landmark
(the eye is never black); the head in front of it counts as body without colour, so percentages are of the whole
fish. Without an eye landmark the line is 3.5 mm behind the snout (flag `colour_start_from_snout`); without either,
the whole fish is measured (flag `colour_start_missing`). The landmark and distances are in the profile
(`colour_start`). The line is drawn in yellow, dotted, in 5 Results.

**Size only.** When Settings says size only on every fish, the screen says so and **Measure colours on every fish**
switches it back.

**Where the models are.** In the project folder, `models\colours_v1.npz` + `colours_v1.json` (date, annotator,
photos, pixels per class, `started_from`); every training adds v2, v3… A bar shows the training progress. Models
trained from 0.12 on also keep up to 10,000 of their training pixels per class, so they can be a start for Ritocco.

**Optional classes.** A class marked optional in the profile (for the guppy male: iridescent) is measured only when
Settings > What to measure > *Measure the colour class …* is ticked; new projects start with it off. When off it has
no brush, its pixels count as body without colour and its CSV columns are `NA`.

**Export model… / Import model…** save the model chosen in *Use in the analysis* as one `.zip` file, to use it in
another project or give it to a colleague. Import checks the file (fixed contents, SHA-256 of each part) and that
its colour classes are among those of the project's species profile (a model with only orange, black and body
without colour is fine: paint the other classes and train with Ritocco), then adds it as the next version and puts it in use;
the model keeps where it came from (`imported_from` in its `.json`).

Strokes made in Windows Paint can be imported: save a copy of the photo as `<photo name>_strokes.png` next
to it and paint with Paint's orange (carotenoid), green (black), blue (iridescent), yellow (body without
colour) and purple (reflection), then **Import strokes from Paint**.

### 8 Landmarks
1. The list shows the analysed fish (photos with clicks first, marked ✓). The profile's landmarks are on the
   right in order: for the guppy male *snout, eye, dorsal fin base, gonopodium base, top of the caudal fin base,
   end of the caudal peduncle, bottom of the caudal fin base, tip of the tail*. The top and bottom of the caudal fin
   base are optional: with them the body ends on the arc through these three points (the peduncle ends rounded).
2. Click each landmark on the fish; the highlighted one is next. **S** skips an optional landmark,
   **Backspace** undoes, **N** goes to the next photo. Mouse wheel zooms, right button drags. Clicks are saved
   at once and the **standard length** (snout to end of the caudal peduncle, along the fish) goes into the results
   straight away: no new analysis needed. A landmark clicked outside the outline (e.g. the tail tip on a
   transparent fin) also grows the outline at the next analysis (flag `outline_grown_to_landmarks`).
3. With 10–30 photos clicked, **Train** makes a landmark model (YOLO-pose, `models/landmarks_vN`). **Start from**
   chooses where it starts: the small public `yolo11n-pose.pt` weights (downloaded once and checked against their
   SHA-256), or **Ritocco**: a landmark model of the project, e.g. `landmarks_vanderbijl.zip` from the training kit
   (pre-trained on about 1,600 van der Bijl 2025 photos), imported with **Import model…**. Ritocco refines it with
   your clicks in fewer rounds and with gentle steps, so a few photos adapt it to your setup without undoing what
   it learnt. Landmarks a model was never trained on (the kit's has no dorsal fin base and gonopodium base) are
   not predicted until you click them on some photos and train again. The app trains on the
   processor (its installer does not install the graphics-card version), so give it time; the GuppyTraining kit
   uses an NVIDIA card ([MODELS_AND_TRAINING.md](MODELS_AND_TRAINING.md)). The new model is put in use; run the **Analysis** again to get SL (and
   the landmarks) on every fish. Landmarks the model is unsure of (confidence < 0.5) are left out.
   A bar shows the training rounds; it may stop before the last one when the model no longer improves.
   **Export model… / Import model…** work as in Training (the model's landmarks must be among those of the
   project's profile; missing ones are not predicted). Landmark models contain neural-network weights: import only files from people you trust.
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
| `standard_length_mm` | SL, snout to end of the caudal peduncle landmark, measured along the midline of the outline so it follows a bent fish (`NA` without the landmarks) |
| `landmarks_source` | `manual` (clicked), `model` (landmark model) or `NA` |
| `total_length_mm` | TL = SL + the tail fin (end of the peduncle to the tail tip landmark, along the midline); without those landmarks, the outline length when the whole caudal fin is in it; else `NA` (flag `tail_incomplete`) |
| `total_length_source` | `outline`, `landmarks` or `NA` |
| `mask_length_mm` | length of the outline along the fish |
| `caudal_length_mm` | end of the peduncle to tail tip (landmarks), along the midline |
| `max_depth_mm`, `depth_source` | maximum body depth: across the body at the gonopodium base (`landmarks`) or widest point of the outline (`outline`) |
| `fish_area_mm2`, `body_area_mm2`, `body_area_source` | area of the outline; area without the caudal fin, cut on the arc through the caudal fin base landmarks (or straight across at the peduncle landmark) (`landmarks`) or where the outline narrows (`outline`) |
| `caudal_area_mm2` | outline after the peduncle landmark, only when the tail fin is in the outline |
| `outline_source` | what drew the outline: `detector` (automatic), `manual` (redone in Review) or the outline model (`outline_vN`) |
| `area_<class>_mm2` | area of each colour class (with a colour model); `NA` for an optional class that is not measured (Settings > What to measure) |
| `pct_<class>` | % of the whole fish area, head included (reflections left out); colours are measured only behind the head (see 7 Training) |
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
| `colour_start_from_snout` | no eye landmark: colours measured from a fixed distance behind the snout | |
| `colour_start_missing` | no eye or snout landmark: colours measured on the whole fish, head included | |
| `scale_off_session` | px/mm further from the median of the folder than Settings > Scale > Check | yes |
| `size_off_session` | fish length further from the median of the folder than Settings > Scale > Check | yes |
| `outline_grown_to_landmarks` | a landmark was outside the outline (e.g. the tail tip): the outline was grown to include it | |
| `overexposed` | more than 1% of the fish has max(R,G,B) ≥ 250 | |
| `blurry` | sharpness below 40% of the session median | |
| `sl_landmarks_missing` | landmarks present but snout or end of the peduncle missing: no SL | |
| `landmark_model_error` | the landmark model file changed or is missing: not used | |
| `outline_model_missed` | the outline model found no fish: the automatic detector drew the outline | |
| `outline_model_rejected` | the outline model's outline looked wrong (border, pieces, shape): the automatic detector's was better | |
| `outline_model_error` | the outline model file changed or is missing: automatic detector used | |

## Project folder

```
<project>/
  project.json            settings and photo folders
  colour_reference.csv    colour reference (Patch,Name,R,G,B)
  results/<photo>.json    everything measured on each photo (pixels)
  masks/<photo>.png       fish outline; <photo>_manual.png = outline redone in Review; <photo>_<part>.png =
                          parts drawn by an outline model (e.g. caudal)
  check/<photo>.jpg       outline and measures drawn on the fish; <photo>_colours.jpg = colour classes
  training/strokes/       painted strokes (one PNG per photo, value = brush number)
  training/images/        colour-corrected photos used for training (cache)
  models/colours_vN.npz   colour models (+ .json with date, annotator, photos, pixels per class)
  scale/<photo>.json      line drawn by hand (ends in photo pixels, mm); scale/folders.json = px/mm per folder
  landmarks/<photo>.json  landmarks clicked by hand (photo pixels, annotator, date)
  models/landmarks_vN/    landmark models (+ landmarks_vN.json with landmarks, photos, SHA-256 of the weights)
  models/outline_vN/      imported outline models (+ outline_vN.json with parts, source, SHA-256 of the weights)
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
- Working with an AI assistant: paste [`LLM_PROMPT.md`](LLM_PROMPT.md) into the chat first (coding agents load it
  through `AGENTS.md` / `CLAUDE.md`). Its *Current state* block is generated from the code by
  `python tools/update_llm_prompt.py`; the `llm-prompt` workflow checks it on pull requests and regenerates it on
  every push to `main`.
- Dependencies: edit `requirements.in`, then
  `uv pip compile requirements.in --python-platform windows --python-version 3.12 --generate-hashes -o requirements-win.txt`.

## What is missing

- Separate body parts (caudal, dorsal, gonopodium): annotation screen and model next.
- Spectra estimated from the photos (module ready, not yet in the app).
- Update check from GitHub releases (off while the repository is private).
- RAW photos (now JPEG/PNG/TIFF): linear demosaicing before the colour correction.

## License

GNU Affero General Public License v3.0 or later (see `LICENSE`), as required by ultralytics.

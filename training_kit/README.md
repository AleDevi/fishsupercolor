# Guppy Colour training kit

Pre-trains two Guppy Colour models on the photos and annotations of **van der Bijl et al. 2025**
(*Nature Ecology & Evolution*; photos and annotations on FRDR, doi
[10.20383/103.01281](https://doi.org/10.20383/103.01281), CC BY 4.0, cite the paper when you use the models):

| Model | What it learns | From which annotation | File in `results` |
| --- | --- | --- | --- |
| Outline (YOLO-seg) | the fish in the whole photo, as two parts: **body** and **caudal fin** (also its transparent part) | `extracted_fish` (body + whole caudal fin), split at the line between landmarks 3 and 4 (start of the caudal fin) | `outline_vanderbijl.zip` |
| Landmarks (YOLO-pose) | **snout**, **eye**, **end of the caudal peduncle**, **tail tip**, so SL and TL | `fixed_landmarks` (1 eye, 2 snout, 3-4 start of the caudal fin: their midpoint is the end of the peduncle); tail tip = point of the outline farthest from the snout | `landmarks_vanderbijl.zip` |

The dorsal fin and the gonopodium are not in the van der Bijl annotations: the landmark model has the
profile's `dorsal_base` and `gonopodium_base` but never saw them, so it does not predict them. Teach them in
Guppy Colour with your own clicks. Colour classes are trained in Guppy Colour as before (van der Bijl has no
iridescent class); their orange and black masks are used here only to check the placement (see below).

It runs without Guppy Colour and without Claude, on any Windows PC with Python 3.12. An NVIDIA card makes it
practical (hours); on the processor only the short test is reasonable.

## Folder layout

```
GuppyTraining\
  install_training.bat      double-click first
  start_training.bat        double-click to train (hours)
  quick_test.bat            a few minutes on a few photos, to check that the kit runs on this PC
  gctrain\                  the training code
  app\                      Guppy Colour itself (same preprocessing and model format as the app)
  data\                     put the van der Bijl files here:
    photo_database.csv
    extracted_fish.zip  carotenoid_coloration.zip  melanic_coloration_v2.zip  fixed_landmarks.zip
    foto_2000\              the photos (any subfolders), e.g. extracted with estrai_foto
    other_photos\           optional: photos of other setups, only drawn in the report
```

The zips are read in place: do not unzip them. They may also sit in a subfolder of `data` (e.g. a copy of the whole
van der Bijl folder). Unzip the kit itself to `C:\`: it makes the folder `C:\GuppyTraining`.

## What `start_training.bat` does

1. **Prepare** (`gctrain/prepare.py`, `gctrain/vdb.py`): each van der Bijl crop is found in its original photo
   (SIFT matches + RANSAC similarity transform; a match is rejected with fewer than 15 inliers or a mean
   grey-level difference above 30) and its annotations are carried onto the photo. Photos are split **by fish**:
   about 80% training, 10% validation, 10% test (never trained on). `work/photos.csv` lists every photo with the
   match quality and, as a check of the placement, the orange % we get against their `car_perc`.
2. **Outline model**: YOLO11s-seg from the ultralytics weights (SHA-256 checked), on whole photos (long side
   1280 px), with colour, scale, rotation and mirror augmentation. A second round adds the 15% of training photos
   it gets most wrong and trains a little more; it is kept only if it does better on the validation photos.
3. **Landmark model**: YOLO11n-pose (the same base as the app) on crops made exactly as the app makes them.
4. **Check** (`gctrain/evaluate.py`) on the test photos: overlap (IoU) of fish, body and caudal fin with
   theirs; landmark errors, SL and TL errors as % of their SL, also apart for the photos whose landmarks they
   placed by hand (the fairest check); the 12 worst photos drawn in `report.html`.
   Their annotations are mostly made by their own model: the numbers say how close we are to them.
5. **Model files** in `results\`, in the layout of Guppy Colour's Export model button.

If it stops (power cut, PC restarted), double-click `start_training.bat` again: finished steps are skipped and
an interrupted training resumes from its last round.

## Using the results in Guppy Colour

- `landmarks_vanderbijl.zip`: Landmarks screen, Import model. Then refine it with your clicks.
- `outline_vanderbijl.zip`: needs a Guppy Colour version that imports outline models (planned).

## Security

Packages are pinned with SHA-256 hashes (`app/requirements-win.txt`; with an NVIDIA card
`requirements-train-cuda-win.txt`, the same list with the CUDA 13.0 builds of torch and torchvision from
download.pytorch.org). The starting weights are downloaded from the ultralytics GitHub release v8.4.0 and checked
against their SHA-256 before they are opened. Nothing is written outside the kit folder and `~/.guppycolor`.

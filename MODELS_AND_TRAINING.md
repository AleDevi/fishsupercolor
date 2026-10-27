# Models and training: where they are, how to load them, how to train on each PC

One page for every model Guppy Colour uses: which ones exist, where they are saved, where to load them in the
app, where to find the ones already trained, and how to train new ones on the app or with the **GuppyTraining**
kit. Updated with every release (last update: app 0.12.0, GuppyTraining 0.2.0).

- [In short: what to use today](#in-short-what-to-use-today)
- [The four kinds of model](#the-four-kinds-of-model)
- [Where the models are saved](#where-the-models-are-saved)
- [Loading a model: which file goes where](#loading-a-model-which-file-goes-where)
- [Models already trained, and where to find them](#models-already-trained-and-where-to-find-them)
- [Which PC does what](#which-pc-does-what)
- [Training in the app (any PC)](#training-in-the-app-any-pc)
- [Training with GuppyTraining (PC with an NVIDIA card)](#training-with-guppytraining-pc-with-an-nvidia-card)
- [Moving models between projects and PCs](#moving-models-between-projects-and-pcs)
- [Is the model good?](#is-the-model-good)
- [Safety](#safety)

## In short: what to use today

| Step | Use | Where in the app |
| --- | --- | --- |
| Fish outline | *Automatic* (YOLOE + SAM 2.1). The van der Bijl outline model is better on photos like theirs, worse on graph paper with closed fins: try both on a few photos | 2 Settings > Fish outline |
| Landmarks (SL, TL, depth) | Import `landmarks_vanderbijl.zip`, click the landmarks on 10–30 of your photos, **Train** with **Start from** = the imported model (Ritocco) | 8 Landmarks |
| Colour classes | Your own model, trained with brush strokes; when the van der Bijl orange/black model is ready, import it and refine it with your strokes (Ritocco) | 7 Training |
| Iridescent | Optional: off in new projects. Switch it on only if you need it, then paint it in Training | 2 Settings > What to measure |

After changing a model, run **3 Analysis** again.

## The four kinds of model

| Kind | What it does | Who makes it | Chosen in |
| --- | --- | --- | --- |
| Base models | find the fish (YOLOE), draw its outline (SAM 2.1), understand the word "fish" (MobileCLIP), read labels (OCR), starting point for landmark training (`yolo11n-pose.pt`) | public, downloaded once by the app and checked | nothing to choose: always there |
| Colour model `colours_vN` | colour class of every pixel of the fish (orange, black, iridescent… as in the profile) | you, with brush strokes | 7 Training > *Use in the analysis* |
| Landmark model `landmarks_vN` | places the profile's landmarks (snout, eye, end of the caudal peduncle, tail tip…) | you, with clicks; or GuppyTraining | 8 Landmarks > *Use in the analysis* |
| Outline model `outline_vN` | draws the fish as body + caudal fin, instead of the automatic detector | GuppyTraining (not trained in the app yet) | 2 Settings > *Fish outline* |

`N` is a version number: every training or import adds the next one and nothing is overwritten.

## Where the models are saved

| What | Folder |
| --- | --- |
| Base models | `C:\Users\<you>\.guppycolor\models\` (`yoloe-11l-seg.pt`, `sam2.1_b.pt`, `mobileclip_blt.ts`, `yolo11n-pose.pt`, `ocr\`) |
| Colour models | `<project folder>\models\colours_vN.npz` + `colours_vN.json` |
| Landmark models | `<project folder>\models\landmarks_vN\weights\best.pt` + `models\landmarks_vN.json` |
| Outline models | `<project folder>\models\outline_vN\best.pt` + `models\outline_vN.json` |
| Exported models | where you save them with **Export model…** (one `.zip` each) |
| GuppyTraining results | `<kit folder>\results\` (e.g. `C:\GuppyTraining\GuppyTraining\results\`) |

Each project keeps its own models: a model trained in project A is not seen by project B until you export it
from A and import it into B. The `.json` next to each model says when and from what it was made (date,
annotator, photos, `imported_from`, `started_from`, SHA-256 of the weights). Do not rename or edit these
files: the app checks the SHA-256 and refuses a changed model.

## Loading a model: which file goes where

Every model travels as one `.zip` (made by **Export model…** or by GuppyTraining). The zip knows its kind, so
it has to be imported on the screen of that kind:

| File | Kind | Import it in | Then |
| --- | --- | --- | --- |
| `colours_*.zip`, e.g. `colours_vanderbijl.zip` | colour | 7 Training > **Import model…** | it is put in use; check it with *Show what the model sees*, then refine it (Ritocco, below). It may have only some classes (orange, black, body without colour) |
| `landmarks_*.zip`, e.g. `landmarks_vanderbijl.zip` | landmarks | 8 Landmarks > **Import model…** | it is put in use; for your photos, refine it (Ritocco, below) |
| `outline_*.zip`, e.g. `outline_vanderbijl.zip` | outline | 2 Settings > Fish outline > **Import outline model…** | it is put in use; *Automatic* in the same list switches back |

A zip imported on the wrong screen is refused with a message saying which screen to use. Import also checks that
the colour classes, landmarks or body parts of the model are those of the project's profile.

## Models already trained, and where to find them

| Model | Trained on | Where | Notes |
| --- | --- | --- | --- |
| `landmarks_vanderbijl.zip` | about 1,570 van der Bijl 2025 photos (live males in a tank), GuppyTraining 0.1.1, 26 Oct 2026 | work PC: `Experiments\Padova\Guppy colouration automation\Risultati Training\results` | snout, eye, end of the caudal peduncle, tail tip; no dorsal fin base, no gonopodium base (click those yourself). On their test photos SL error about 1.5%; on our graph-paper photos the snout/eye/peduncle are good on about half: use it as the start for Ritocco |
| `outline_vanderbijl.zip` | same photos | same folder | body and caudal fin; on their photos overlap 0.98; on ours it often misses the closed, transparent caudal fin |
| `colours_vanderbijl.zip` | van der Bijl orange and black masks (GuppyTraining 0.2.0) | not made yet: run GuppyTraining 0.2.0 on the lab PC (below); it will land in the same `results` folder | orange, black and body without colour, no iridescent; their photos have no colour chart, so refine it on yours |
| your colour and landmark models | your photos | `models\` of each project | export them to reuse them |

The report of that training (`report.html`) and the test on our photos (`prova_foto_Ale.html`) are in the same
`Risultati Training` folder. Cite van der Bijl et al. 2025 (CC BY 4.0) when you publish results made with the
`*_vanderbijl` models.

## Which PC does what

| PC | Graphics card | Use it for |
| --- | --- | --- |
| Work PC (app + Claude) | none usable for training | the app: analysis, review, clicks and brush strokes, colour training, landmark Ritocco (slower without GPU: let it run) |
| Lab PC with the RTX 4070 | NVIDIA, 12 GB | GuppyTraining (about 1.5 h for the full run); the app works there too |
| Home PC | GT 750M, too old: processor only | the app; GuppyTraining only for `quick_test.bat` |

Colour training never needs a graphics card. The app itself runs on the processor on every PC (its installer
installs the processor version of the neural-network library), so landmark training in the app is slow but works
everywhere: start it and let it run. Only GuppyTraining installs the version that uses the NVIDIA card.

## Training in the app (any PC)

**Colour classes** (7 Training): paint strokes of each class on 5–10 fish, **Train**, look at *Show what the
model sees*, add strokes where it is wrong, train again. **Start from** works as for landmarks (Ritocco): choose an
imported colour model and your strokes count 3 times more than its pixels. Only models that keep their training
pixels can be a start: those trained or exported with app 0.12 or later. Optional classes (iridescent) have a
brush only when they are switched on in Settings. Strokes and colours count only inside the fish outline, so fix
the outline (tail included) in 4 Review first. Details in the README, section *7 Training*.

**Landmarks** (8 Landmarks):

1. Click the landmarks on 10–30 photos (clicks count at once for SL, without a new analysis).
2. **Start from**: `yolo11n-pose.pt` to start from scratch, or an imported landmark model to refine it
   (**Ritocco**: fewer rounds, gentle steps, so a few of your photos adapt it to your setup without undoing what
   it learnt). Ritocco needs the model to have the same landmarks as the project's profile.
3. **Train**. The new model `landmarks_vN` is put in use. Run the Analysis again.
4. Landmarks the model was never trained on are not predicted until you click them on some photos and train
   again.

**Outline**: not trained in the app yet (planned). Today an outline model comes only from GuppyTraining.

## Training with GuppyTraining (PC with an NVIDIA card)

GuppyTraining is a separate kit (`GuppyTraining_<version>.zip`) that pre-trains the outline and landmark models
on the van der Bijl 2025 photos and annotations, without Claude and without the app's window.

You need: Windows, Python 3.12 from python.org ("Add python.exe to PATH" ticked), an up-to-date NVIDIA driver,
internet the first time (about 3 GB of packages), about 30 GB free.

1. Unzip the kit to `C:\`: it makes `C:\GuppyTraining\GuppyTraining`.
2. Put the van der Bijl files in its `data\` folder: `photo_database.csv`, `extracted_fish.zip`,
   `carotenoid_coloration.zip`, `melanic_coloration_v2.zip`, `fixed_landmarks.zip` (leave the zips zipped) and
   the photos in `data\foto_2000\`. Optional: other photos in `data\other_photos\`, only drawn in the report.
3. Double-click `install_training.bat`. At the end it must say "GPU OK".
   **Updating the kit** (e.g. 0.1.1 to 0.2.0): unzip the new version over the old folder, keep `data\` and
   `work\`, run `install_training.bat` again, then `start_training.bat`: only the new steps run.
4. Double-click `quick_test.bat` (a few minutes) to check the kit runs on this PC.
5. Double-click `start_training.bat` and leave the PC on (about 1.5 h on the RTX 4070). If it stops, double-click
   it again: it resumes, and steps already finished are skipped.
6. The results are in `results\`: `landmarks_vanderbijl.zip`, `outline_vanderbijl.zip`, `colours_vanderbijl.zip`
   (from 0.2.0), `report.html` (with % orange and % black against van der Bijl's own percentages) and
   `colour_test_results.csv`. Copy the
   folder to a USB stick, put it on the work PC in `Guppy colouration automation\Risultati Training\`, and import
   the zips as in [Loading a model](#loading-a-model-which-file-goes-where).

## Moving models between projects and PCs

1. In the project that has the model: choose it in *Use in the analysis* (or in Settings for outlines) and
   **Export model…**: one `.zip`.
2. Copy the `.zip` (USB stick, shared folder, e-mail).
3. In the other project, on any PC: **Import model…** on the screen of its kind. The app checks the file and adds
   it as the next version.

The base models need no moving: each PC downloads them once.

## Is the model good?

- **Colours and sizes**: 7 Training > **Compare with measures made by hand** (a table with the fish id and e.g.
  `OrangeT`, `BlackT`, `Totallength`, `Area`): mean difference, mean absolute error, correlation, Bland-Altman
  limits.
- **Landmarks**: look at the points in 8 Landmarks and at the SL/TL in 5 Results; compare SL with hand measures as
  above.
- **Outline**: 4 Review and the CSV column `outline_source`; the flags `outline_model_missed` and
  `outline_model_rejected` count the photos where the outline model was not used.
- **GuppyTraining**: `report.html` gives the errors on van der Bijl photos it never trained on.

## Safety

Colour models are plain numbers (`.npz`) and are safe to share. Landmark and outline models contain
neural-network weights: import only files you made yourself or got from people you trust. Every import is checked
against the SHA-256 written in the zip, and the base models against fixed SHA-256 values in the app.

# Sample photos

One real photo for each photo setup used so far in the guppy experiments, for tests (local and GitHub Actions)
and for trying the app. The files are unmodified copies of the originals (same pixels and EXIF); only the
file names are new, neutral ids. The handwritten labels inside the photos still show the original fish ids.

All photos: Canon EOS 450D, JPEG, 4272 x 2848 px, auto white balance, no flash.
Details per photo are in [`samples.csv`](samples.csv).

| File | Setup | Sex | Colour chart | Scale | EXIF orientation |
|---|---|---|---|---|---|
| sample_01.jpg | Close-up, polyandry/heatwave experiment block 1 (2023) | male | no | graph paper (1 mm) | normal |
| sample_02.jpg | Close-up, block 2 (2023) | male | no | graph paper (1 mm) | normal |
| sample_03.jpg | Close-up, block 3 (2023), darker exposure | male | no | graph paper (1 mm) | normal |
| sample_04.jpg | Wide shot, males block 1 (2023), fish small in frame | male | yes (24 patches) | graph paper | rotated 90° (portrait) |
| sample_05.jpg | Wide shot, males block 2 (2023), fish vertical | male | yes (24 patches) | graph paper | normal |
| sample_06.jpg | Wide shot, male photos 2024 (orange graph paper) | male | yes (24 patches) | graph paper (orange) | normal |
| sample_07.jpg | Offspring photos 2025 (spectra project), fish in a small tank | male (young) | yes (24 patches) | graph paper | rotated 180° |
| sample_08.jpg | Offspring photos 2025, female (size only) | female | yes (24 patches) | graph paper | rotated 180° |

Notes
- The block 1-3 close-ups have no colour chart and use auto exposure, so brightness differs between them.
- sample_04, sample_07 and sample_08 carry an EXIF rotation: the app must apply it before analysis.
- Fish in the offspring photos are small and pale; good test cases for the fish detector.
- These photos are not part of the app release zip (`tests/data` is export-ignored).

# DSAV360 synchronisation test

Measured playback rate (displayed frame vs data time, 97 test recordings with measured-gaze windows): median 60.00 fps (5-95%: 59.99-60.00). Median start frame by video: 1: 0, 2: 1, 3: 300, 4: 0, 5: 240, 6: 0, 7: 0, 8: 60, 9: 0, 10: 420.

Windows with measured gaze targets: 3774 (videos 3/5/8/10 with a start offset: 1039). Model: behaviour-aware, lambda = 20, 3 seeds (mean ± s.d.). NSS: standardised saliency at the gaze position (higher = better aligned).

| condition | step 10 | step 20 | step 10 (offset videos) | step 20 (offset videos) | NSS all | NSS offset videos |
|---|---|---|---|---|---|---|
| pipeline: 60 fps, no offset | 15.61 ± 0.06 | 23.82 ± 0.11 | 16.77 | 25.50 | 1.425 | 1.308 |
| measured mapping (fit) | 15.59 ± 0.05 | 23.81 ± 0.10 | 16.72 | 25.45 | 1.466 | 1.457 |
| 30 fps + measured offset | 15.64 ± 0.07 | 23.85 ± 0.12 | 16.85 | 25.58 | 1.373 | 1.246 |
| 45 fps + measured offset | 15.63 ± 0.06 | 23.84 ± 0.12 | 16.80 | 25.52 | 1.385 | 1.290 |
| 55 fps + measured offset | 15.62 ± 0.06 | 23.82 ± 0.10 | 16.80 | 25.52 | 1.411 | 1.342 |
| 60 fps + measured offset | 15.59 ± 0.05 | 23.81 ± 0.10 | 16.72 | 25.45 | 1.466 | 1.457 |
| 65 fps + measured offset | 15.61 ± 0.06 | 23.82 ± 0.10 | 16.78 | 25.49 | 1.419 | 1.348 |
| 75 fps + measured offset | 15.60 ± 0.05 | 23.82 ± 0.10 | 16.78 | 25.52 | 1.395 | 1.313 |
| 90 fps + measured offset | 15.63 ± 0.05 | 23.83 ± 0.09 | 16.82 | 25.53 | 1.389 | 1.320 |
| 120 fps + measured offset | 15.63 ± 0.06 | 23.85 ± 0.10 | 16.83 | 25.54 | 1.381 | 1.300 |
| cached saliency (mp4 clips, as evaluated) | 15.59 ± 0.06 | 23.81 ± 0.11 | 16.73 | 25.46 | – | – |

# E8 DSAV360 data checks

Processed subset: videos 1–10 (of 85), 88 participants, 684 recordings (train 478 / val 103 / test 103; only test is used, zero-shot). Stereo viewers among test recordings: 50.5%.

## Zero-shot error by viewing condition

| model | condition | N | step10 | step20 | HMD step10 | HMD step20 |
|---|---|---|---|---|---|---|
| e1_gaze_s0 | mono | 2448 | 20.74 | 29.18 | 25.70 | 34.55 |
| e1_gaze_s0 | stereo | 2496 | 18.74 | 27.32 | 24.57 | 34.10 |
| e2_dual20_s0 | mono | 2448 | 20.78 | 29.16 | 25.70 | 34.55 |
| e2_dual20_s0 | stereo | 2496 | 18.84 | 27.31 | 24.57 | 34.10 |
| e1_dual100_s0 | mono | 2448 | 20.68 | 28.94 | 25.70 | 34.55 |
| e1_dual100_s0 | stereo | 2496 | 18.83 | 27.12 | 24.57 | 34.10 |
| base_hmd_cv | mono | 2448 | 19.84 | 29.53 | 25.70 | 34.55 |
| base_hmd_cv | stereo | 2496 | 18.22 | 27.76 | 24.57 | 34.10 |

## Step-0 head–gaze offset at prediction time (great-circle, deg)

| dataset | N windows | mean | median | 90th pct |
|---|---|---|---|---|
| eht | 11432 | 11.27 | 10.17 | 19.09 |
| dsav | 4944 | 16.90 | 13.00 | 28.09 |

## Stereo vs mono viewers, behaviour-aware lambda=20 (3 seeds)

| horizon | mono | stereo |
|---|---|---|
| step 10 | 20.75 ± 0.04 | 18.83 ± 0.05 |
| step 20 | 29.09 ± 0.09 | 27.28 ± 0.06 |

## Head–gaze longitudinal offset, all recorded test frames (deg)

| dataset | frames | mean signed | mean absolute |
|---|---|---|---|
| eht | 214350 | 1.59 | 7.53 |
| dsav | 92700 | 1.04 | 13.71 |
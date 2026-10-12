# E8 DSAV360 data checks

Processed subset: videos 1–10 (of 85), 88 participants, 684 recordings (train 478 / val 103 / test 103; only test is used, zero-shot). Stereo viewers among test recordings: 50.5%.

## Zero-shot error by viewing condition

| model | condition | N | step10 | step20 | HMD step10 | HMD step20 |
|---|---|---|---|---|---|---|
| e1_gaze_s0 | mono | 1780 | 15.40 | 23.38 | 20.78 | 29.08 |
| e1_gaze_s0 | stereo | 1994 | 15.65 | 24.33 | 22.23 | 31.94 |
| e2_dual20_s0 | mono | 1780 | 15.48 | 23.40 | 20.78 | 29.08 |
| e2_dual20_s0 | stereo | 1994 | 15.75 | 24.30 | 22.23 | 31.94 |
| e1_dual100_s0 | mono | 1780 | 15.31 | 23.04 | 20.78 | 29.08 |
| e1_dual100_s0 | stereo | 1994 | 15.78 | 24.16 | 22.23 | 31.94 |
| base_hmd_cv | mono | 1780 | 14.41 | 23.89 | 20.78 | 29.08 |
| base_hmd_cv | stereo | 1994 | 15.23 | 25.00 | 22.23 | 31.94 |

## Step-0 head–gaze offset at prediction time (great-circle, deg)

| dataset | N windows | mean | median | 90th pct |
|---|---|---|---|---|
| eht | 11432 | 11.27 | 10.17 | 19.09 |
| dsav | 3774 | 12.97 | 12.10 | 22.76 |

## Stereo vs mono viewers, behaviour-aware lambda=20 (3 seeds)

| horizon | mono | stereo |
|---|---|---|
| step 10 | 15.44 ± 0.04 | 15.73 ± 0.08 |
| step 20 | 23.31 ± 0.10 | 24.25 ± 0.12 |

## Head–gaze longitudinal offset, all recorded test frames (deg)

| dataset | frames | mean signed | mean absolute |
|---|---|---|---|
| eht | 214350 | 1.59 | 7.53 |
| dsav | 80632 | 1.18 | 10.31 |
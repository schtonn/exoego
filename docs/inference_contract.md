# Layered pipeline input contract

Updated: 2026-09-30

The exported `manifest.json` is the source of truth. The terminal fusion loader
rejects missing, stale, or mismatched contracts instead of trusting an
experiment name.

## Main reported configuration

| Information | Used for inference? | Source |
| --- | --- | --- |
| Four synchronized exo RGB-D streams and calibration | yes | H2O cam0–cam3 |
| Ego RGB-D at frame 0 | yes | H2O cam4 |
| Ego camera pose at frame 0 | yes | H2O cam4 ground truth |
| Future ego camera pose | no | estimated from exo face RGB-D |
| Object pose | yes | H2O annotation; upper-bound input |
| Future ego RGB | no | visualization and evaluation only |

The main configuration is therefore an anchored, annotated-object experiment,
not a pure exo deployment result. Reduced-input exports have separate protocol
names for no anchor, estimated initial mount, and no object annotation.

## Motion and causality

Head motion is estimated from nine face landmarks sampled in exo depth, fitted
with RANSAC, smoothed, and applied to the initial camera pose. The default
rotation scale of 0.5 was chosen during development and must remain frozen for
subject-4 testing. New head summaries record inlier residuals and a pose
confidence; terminal fusion uses that confidence when limiting corrections to
transported pixels.

`--causal-motion-masks` builds the exo depth background using only current and
past frames. Without it, the historical default uses the whole clip. ProPainter
and the ±2-frame hand-boundary cleanup are offline, so an output using them is
not fully causal even when the motion mask is causal.

## Composition

The layered renderer keeps ego-anchor transport, current exo, historical exo,
generated background, hand/arm, and object evidence separate. The new terminal
fusion uses a per-pixel correction budget:

- full correction in explicit repair pixels;
- smaller corrections on transported interiors;
- larger but bounded corrections at source seams and hand/object boundaries;
- reliability is reduced when exo head-pose confidence is low.

The learned base/proposal mix is convex and its residual is bounded. Legacy
checkpoints retain the old extrapolating formula only when loaded explicitly as
legacy checkpoints.

## Evaluation rules

Subjects 1–2 are training, subject 3 is validation, and subject 4 is test.
Training and validation loaders enforce these roles. Hyperparameters and
checkpoints are selected on subject 3 and then frozen. Report:

- copy-first-frame and motion-warped-first-frame baselines in the same harness;
- full-frame and provenance-region L1, PSNR, and temporal delta error;
- authorized/editable pixel fractions;
- hand/object IoU when GT masks are supplied;
- pose-noise curves and variation across scenes and seeds.

At present only one of the planned eight subject-4 scenes has completed the
full layered + ProPainter pipeline. No aggregate subject-4 claim should be made
until all eight finish.

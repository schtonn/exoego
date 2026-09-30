# Concise experimental status

## Protocol

The main H2O validation uses 64-frame clips at 15 fps. Reconstruction metrics
exclude frame 0 because protocols with an egocentric anchor copy that frame by
definition. The terminal fusion model has 49,788 parameters and may only modify
authorized repair pixels; reliable hand/arm and object pixels are locked.

## Eight-scene input ablation

Equal-weight means over eight subject-3 **validation** scenes. These scenes were
also used for checkpoint selection and some hyperparameter choices, so the
numbers below are development results rather than final test results:

| Input contract | Full-frame L1 |
| --- | ---: |
| 4 exo + ego first frame | 0.074624 |
| 4 exo, no ego first frame | 0.117922 |
| cam0 + ego first frame | 0.081118 |
| cam0, no ego first frame | 0.182746 |
| 4 exo, no first frame, no supplied head-camera relation | 0.183327 |

The first egocentric frame mainly supplies target-view background appearance.
Multi-view exocentric input improves coverage and state estimation. The initial
head-camera relationship is an independent and important source of information.

## Hand/arm contamination repair

Broad motion RGB-D was previously allowed to become locked arm texture. On the
`h2_3` pilot, enforcing metric skeleton proximity reduced blue table-texture
contamination inside the arm from 3.13% to approximately zero. Missing target
limb support is now exported as a separate dynamic-completion mask.

The current variant uses one ring of limb densification instead of two and a
tighter secondary-view appearance radius (`0.24` instead of `0.38`). Relative
to the preceding strict-completion version, dynamic-region L1 improved in all
five input protocols by approximately 0.84% to 2.62% on `h2_3`.

## Limitations

- Generic video inpainting can extend background instead of synthesizing an
  unseen arm, especially for single-view/no-anchor inputs.
- RGB-D point splatting still produces residual texture discontinuities. A
  dense skeleton-conditioned limb mesh or learned local feature renderer is the
  next intended replacement.
- Object pose currently uses H2O annotations as an upper bound.
- Camera/head estimation and final generation have not yet been validated at
  Ego-Exo4D scale.

## Reviewer-driven checks (2026-09-30)

On validation scene `subject3/h2/3`, the same fixed-axis camera-pose rotation
perturbation gives the following layered-output curve (all other settings are
fixed):

| Added rotation | Full-frame L1 | Observed fraction |
| ---: | ---: | ---: |
| 0° | 0.06911 | 0.9479 |
| 1° | 0.06975 | 0.9390 |
| 3° | 0.11428 | 0.9166 |
| 5° | 0.15413 | 0.8958 |

The 3° error raises L1 by 65.3%, and the 5° error by 123.0%. This confirms that
pose robustness is a primary bottleneck, not a documentation issue. This is one
scene and one perturbation axis; more scenes and noise seeds remain necessary.

The first frozen subject-4 scene (`subject4/h2/3`) has now been run end to end:

| Method | L1 | temporal delta L1 |
| --- | ---: | ---: |
| Copy ego first frame | 0.10530 | 0.01022 |
| Warp ego first frame, copy-fill holes | 0.09413 | 0.01734 |
| Layered geometry, no completion | 0.09044 | 0.02395 |
| Constrained ProPainter composition | **0.08656** | 0.02203 |

This is one test scene, not the final eight-scene test. Terminal fusion has not
yet been retrained under the new reliability-weighted contract. LPIPS, manual
pixel-mask IoU, seed variance, and the remaining subject-4 scenes are still
outstanding. Against evaluation-only masks projected from annotated hand joints
and a depth-filtered object CAD silhouette, this scene obtains hand/arm IoU
0.330 and object IoU 0.293. These are proxy masks rather than manual pixel
labels, but the low overlap confirms that mask reliability remains a major
problem. The evaluator reports provenance-region errors, authorized area,
optional reference-mask IoU, and identifies predicted-mask metrics explicitly.

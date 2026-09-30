# Concise experimental status

## Protocol

The main H2O validation uses 64-frame clips at 15 fps. Reconstruction metrics
exclude frame 0 because protocols with an egocentric anchor copy that frame by
definition. The terminal fusion model has 49,788 parameters and may only modify
authorized repair pixels; reliable hand/arm and object pixels are locked.

## Eight-scene input ablation

Equal-weight means over eight held-out subject-3 scenes:

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


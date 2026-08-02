"""Grad-CAM for the segmentation head — target-specific saliency per level.

Why this exists alongside ``panels.channel_heat``: that function renders
``mean|activation|`` across channels, an *unsigned magnitude* map. It shows
what a layer responds to (texture, frequency), not what the network decided,
which is exactly why such panels never look like the predicted mask. Grad-CAM
answers the other question — which spatial locations drove *this* prediction.

Method: Seg-Grad-CAM (Vinogradova et al., 2020). A scalar score is formed by
summing ``mask_logit`` over a region of interest, then gradients of that score
are taken with respect to the tapped feature maps. Channel weights are the
spatially-averaged gradients; the CAM is the ReLU of their weighted sum.

Implementation note: ``CHDNet.forward(..., return_intermediates=True)`` hands
back the actual graph tensors, so ``torch.autograd.grad`` can be called on
them directly — **no forward or backward hooks are needed anywhere**. This
keeps the model file untouched.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chd.viz.panels import normalize01

#: Per-level taps, finest-to-coarsest lists of 4 tensors each.
LEVEL_TAPS = ("aer", "osneck", "sfa", "backbone")

#: Module-boundary progression at a single level: (display label, intermediates key).
#: FDM's high-frequency branch is the camouflage-relevant half of its split,
#: so that is the one shown in the progression.
PROGRESSION_TAPS = (
    ("Backbone", "backbone"),
    ("+FDM (HF)", "fdm_hf"),
    ("+SFA", "sfa"),
    ("+OSNeck", "osneck"),
    ("+AER", "aer"),
    ("Decoder", "decoder_levels"),
)

FALLBACK_TAP = "backbone"


def cam_score(
    model, outputs: dict, target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> torch.Tensor:
    """Scalar to differentiate: ``mask_logit`` summed over a region of interest.

    ``target="pred"`` explains the model's own decision, which is what a
    "why was this mask highlighted" figure needs. When the prediction is empty
    (every negative image, and any missed target) summing over it would give a
    constant zero with no gradient, so the score falls back to the ``topk``
    highest logits — still the model's own evidence, just unthresholded.
    """
    logit = outputs["mask_logit"]

    if target == "all":
        return logit.sum()

    if target == "gt":
        if gt is None:
            raise ValueError("target='gt' requires a gt tensor")
        region = (gt > 0.5).to(logit.dtype)
    else:
        region = (model.predict_mask(outputs) > 0.5).to(logit.dtype)

    if float(region.sum()) == 0.0:
        flat = logit.flatten()
        k = min(topk, flat.numel())
        return torch.topk(flat, k).values.sum()
    return (logit * region).sum()


def cams_from(
    acts: list[torch.Tensor], grads: tuple[torch.Tensor, ...], out_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Grad-CAM per (activation, gradient) pair, upsampled to ``out_hw``."""
    cams: list[np.ndarray] = []
    for activation, gradient in zip(acts, grads):
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
        cams.append(normalize01(cam[0, 0].detach().float().cpu().numpy()))
    return cams


def _grad_for(score: torch.Tensor, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """``autograd.grad`` over possibly-duplicated tensors.

    ``PretrainedUNet`` returns ``[decoded] * 4`` for its decoder levels, i.e.
    the same tensor four times. ``autograd.grad`` rejects duplicated inputs, so
    unique tensors are differentiated once and the results expanded back.
    """
    unique: list[torch.Tensor] = []
    positions: list[int] = []
    for tensor in tensors:
        for index, seen in enumerate(unique):
            if seen is tensor:
                positions.append(index)
                break
        else:
            positions.append(len(unique))
            unique.append(tensor)
    grads = torch.autograd.grad(score, unique, retain_graph=True, allow_unused=False)
    return tuple(grads[index] for index in positions)


def _forward_with_intermediates(model, image: torch.Tensor, pose: torch.Tensor) -> dict:
    """Forward pass with gradients enabled — Grad-CAM cannot run under no_grad.

    ``image`` is cloned with ``requires_grad_(True)`` before the forward pass.
    Without this, an architecture with a frozen submodule (e.g.
    ``pretrained_unet --unet-freeze-encoder``) produces intermediate tensors
    with ``requires_grad=False`` whenever none of the ops between the input
    and that tap have a trainable parameter — ``autograd.grad`` then raises
    "One of the differentiated Tensors does not require grad" on that tap.
    Forcing the input to require grad makes every op downstream differentiable
    with respect to it regardless of which parameters are frozen.
    """
    image = image.clone().requires_grad_(True)
    with torch.enable_grad():
        return model(image, pose, return_intermediates=True)


def _resolve_tap(intermediates: dict, tap: str) -> str:
    """Fall back to the backbone when an architecture lacks the requested tap."""
    if intermediates.get(tap):
        return tap
    return FALLBACK_TAP


def grad_cam_levels(
    model, image: torch.Tensor, pose: torch.Tensor, tap: str = "aer",
    target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> tuple[list[np.ndarray], str]:
    """One Grad-CAM per pyramid level from a single forward/backward pass.

    Returns ``(cams_finest_to_coarsest, tap_actually_used)``. The tap is
    returned because it may have been downgraded for an architecture that
    lacks the requested module — the caller needs to label the figure honestly.
    """
    model.zero_grad(set_to_none=True)
    outputs = _forward_with_intermediates(model, image, pose)
    intermediates = outputs["intermediates"]

    used = _resolve_tap(intermediates, tap)
    acts = list(intermediates[used])
    score = cam_score(model, outputs, target=target, gt=gt, topk=topk)
    grads = _grad_for(score, acts)
    return cams_from(acts, grads, tuple(image.shape[-2:])), used


def grad_cam_progression(
    model, image: torch.Tensor, pose: torch.Tensor, level: int = 0,
    target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> list[tuple[str, np.ndarray]]:
    """Grad-CAM at each module boundary for one pyramid level.

    Shows where evidence moves as the forward pass proceeds. This is a
    *within-network* progression, not a cumulative ablation across separately
    trained variants — see the design doc.

    Modules the architecture does not have are skipped rather than faked.
    """
    model.zero_grad(set_to_none=True)
    outputs = _forward_with_intermediates(model, image, pose)
    intermediates = outputs["intermediates"]

    labels: list[str] = []
    acts: list[torch.Tensor] = []
    for label, key in PROGRESSION_TAPS:
        tensors = intermediates.get(key)
        if not tensors or level >= len(tensors):
            continue
        labels.append(label)
        acts.append(tensors[level])

    score = cam_score(model, outputs, target=target, gt=gt, topk=topk)
    grads = _grad_for(score, acts)
    heats = cams_from(acts, grads, tuple(image.shape[-2:]))
    return list(zip(labels, heats))

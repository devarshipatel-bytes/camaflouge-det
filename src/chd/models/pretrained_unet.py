"""Option C: a plain pretrained U-Net baseline, with minimal fine-tuning.

Deliberately NOT a variant of CHDNet — no FDM, no SFA, no OSNeck, no AER,
no pose prior. It is a standard off-the-shelf segmentation network
(``segmentation_models_pytorch``'s U-Net with an ImageNet-pretrained
encoder) wrapped to speak the same input/output contract as CHDNet so the
training loop, losses, metrics and evaluation are shared unchanged.

The point of this baseline is to answer a question the other two options
can't: **how much of our accuracy comes from the custom architecture versus
just from a competent pretrained segmentation model?** If a frozen-encoder
U-Net matches CHDNet, the custom modules aren't earning their complexity;
if CHDNet clearly wins, they are.

"Minor layer fine-tuning" is the default: the pretrained encoder is frozen
(``freeze_encoder=True``) so only the decoder and the small edge/presence
heads train. That is also the right call for the 100-image CAMO-Human set,
where fine-tuning a full 21M-parameter encoder is exactly how you overfit.

The pose tensor is accepted and ignored, so the same DataLoader batch works
for every architecture without special-casing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PretrainedUNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError(
                "--architecture pretrained_unet needs segmentation-models-pytorch. "
                "Install it with:  pip install segmentation-models-pytorch"
            ) from exc

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.net.encoder.parameters():
                param.requires_grad_(False)

        decoder_channels = self._decoder_out_channels()
        # Small auxiliary heads so this baseline produces the same outputs
        # CHDLoss expects (edge + presence), rather than needing a separate
        # loss function just for this architecture.
        self.edge_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        encoder_out = self.net.encoder.out_channels[-1]
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def _decoder_out_channels(self) -> int:
        """Width of the U-Net decoder's final feature map, before the seg head."""
        # smp exposes this differently across versions; probe the segmentation
        # head's input width, which is the one thing guaranteed to be correct.
        head_conv = self.net.segmentation_head[0]
        return head_conv.in_channels

    def train(self, mode: bool = True):  # noqa: D102 - keeps frozen encoder in eval
        super().train(mode)
        if self.freeze_encoder:
            # A frozen encoder must not keep updating BatchNorm running stats,
            # or it silently drifts away from the pretrained statistics even
            # though its weights never change.
            self.net.encoder.eval()
        return self

    def forward(
        self, image: torch.Tensor, pose: torch.Tensor, return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """``pose`` is accepted for interface parity with CHDNet and ignored."""
        del pose
        out_size = image.shape[-2:]

        encoder_feats = self.net.encoder(image)
        decoded = self.net.decoder(encoder_feats)
        mask_logit = self.net.segmentation_head(decoded)
        if mask_logit.shape[-2:] != out_size:
            mask_logit = F.interpolate(mask_logit, size=out_size, mode="bilinear", align_corners=False)

        edge_logit = self.edge_head(decoded)
        if edge_logit.shape[-2:] != out_size:
            edge_logit = F.interpolate(edge_logit, size=out_size, mode="bilinear", align_corners=False)

        presence_logit = self.presence_head(encoder_feats[-1]).squeeze(-1)

        outputs = {
            "mask_logit": mask_logit,
            # CHDLoss expects 4 deep-supervision maps; a plain U-Net has no
            # per-level side outputs, so reuse the single prediction. The
            # side_weights then act purely as an extra weight on the main
            # loss for this architecture, which keeps the loss comparable
            # across all three options without inventing fake side heads.
            "side_logits": [mask_logit] * 4,
            "edge_logit": edge_logit,
            "presence_logit": presence_logit,
        }
        if return_intermediates:
            outputs["intermediates"] = {
                "backbone": list(encoder_feats[1:5]),
                "decoder_levels": [decoded] * 4,
            }
        return outputs

    @staticmethod
    def predict_mask(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        presence = torch.sigmoid(outputs["presence_logit"]).view(-1, 1, 1, 1)
        return torch.sigmoid(outputs["mask_logit"]) * presence

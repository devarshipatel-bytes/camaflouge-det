"""OS-Res2Net-CHDNet: the full camouflaged-human-detection network.

    input 3xHxW
    -> Res2Net-50 backbone                     F1..F4 @ 256/512/1024/2048ch
    -> per-level Conv1x1 reduce                 -> 64ch each
    -> per-level FDM                            -> F_LF, F_HF_hat
    -> per-level SFA                            -> F_SFA
    -> per-level OSBlock (omni-scale neck)      -> F_OS
    -> per-level AER (pose-guided attention)    -> F_tilde
    -> PDC decoder (top-down multiplicative)    -> main mask logit + side feats
    -> heads: side masks (deep supervision), edge, presence gate
    -> output = sigmoid(main mask logit) * sigmoid(presence logit)

Every module is resolution-agnostic (no fixed-size layers anywhere), so
``--img-size`` is purely a data-loading choice, never a model change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chd.models.aer import AER
from chd.models.backbone import FEATURE_CHANNELS, build_backbone
from chd.models.decoder import PDCDecoder
from chd.models.fdm import FDM
from chd.models.heads import EdgeHead, PresenceGate, SideHeads
from chd.models.osblock import OSBlock
from chd.models.sfa import SFA

N_LEVELS = 4


class CHDNet(nn.Module):
    def __init__(
        self,
        backbone: str = "res2net50_26w_4s",
        pretrained: bool = True,
        channels: int = 64,
        os_streams: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained)

        self.reduce = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, channels, kernel_size=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for c in FEATURE_CHANNELS
        ])
        self.fdm = nn.ModuleList([FDM(channels) for _ in range(N_LEVELS)])
        self.sfa = nn.ModuleList([SFA(channels) for _ in range(N_LEVELS)])
        self.osneck = nn.ModuleList([OSBlock(channels, channels, n_streams=os_streams) for _ in range(N_LEVELS)])
        self.aer = nn.ModuleList([AER(channels) for _ in range(N_LEVELS)])

        self.decoder = PDCDecoder(channels)
        self.side_heads = SideHeads(channels, N_LEVELS)
        self.edge_head = EdgeHead(channels)
        self.presence_gate = PresenceGate(in_channels=FEATURE_CHANNELS[-1])

    def forward(self, image: torch.Tensor, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        """``pose``: (B, 17, Hp, Wp), any spatial size — resized per level inside AER."""
        out_size = image.shape[-2:]
        backbone_feats = self.backbone(image)  # [F1, F2, F3, F4], raw channel counts

        tilde = []
        for i, feat in enumerate(backbone_feats):
            reduced = self.reduce[i](feat)
            f_lf, f_hf_hat = self.fdm[i](reduced)
            f_sfa = self.sfa[i](reduced, f_lf, f_hf_hat)
            f_os = self.osneck[i](f_sfa)
            tilde.append(self.aer[i](f_os, pose))

        main_logit, side_feats = self.decoder(*tilde)
        main_logit = F.interpolate(main_logit, size=out_size, mode="bilinear", align_corners=False)
        side_logits = self.side_heads(side_feats, out_size)
        edge_logit = self.edge_head(side_feats[0], out_size)
        presence_logit = self.presence_gate(backbone_feats[-1])

        return {
            "mask_logit": main_logit,
            "side_logits": side_logits,
            "edge_logit": edge_logit,
            "presence_logit": presence_logit,
        }

    @staticmethod
    def predict_mask(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Final camouflaged-human mask: main mask gated by the presence probability."""
        presence = torch.sigmoid(outputs["presence_logit"]).view(-1, 1, 1, 1)
        return torch.sigmoid(outputs["mask_logit"]) * presence

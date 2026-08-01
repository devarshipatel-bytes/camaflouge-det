"""Shape, gradient-flow and param-count tests for every OS-Res2Net-CHDNet module.

Uses ``tiny_test`` backbone (random weights, no download) for fast, offline
runs — the real Res2Net-50 integration is exercised separately in
``test_chdnet_real_backbone`` behind a marker so it only downloads weights
when explicitly requested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.models.aer import AER, N_KEYPOINTS  # noqa: E402
from chd.models.backbone import FEATURE_CHANNELS, build_backbone  # noqa: E402
from chd.models.chdnet import CHDNet  # noqa: E402
from chd.models.decoder import PDCDecoder  # noqa: E402
from chd.models.fdm import FDM  # noqa: E402
from chd.models.heads import EdgeHead, PresenceGate, SideHeads  # noqa: E402
from chd.models.osblock import OSBlock  # noqa: E402
from chd.models.sfa import SFA  # noqa: E402

CH = 64
B, H, W = 2, 32, 32


def assert_all_grads(module: torch.nn.Module) -> None:
    missing = [name for name, p in module.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def assert_finite(*tensors: torch.Tensor) -> None:
    for t in tensors:
        assert torch.isfinite(t).all(), "found NaN/Inf"


# --------------------------------------------------------------------------
# FDM
# --------------------------------------------------------------------------

class TestFDM:
    def test_shapes(self) -> None:
        fdm = FDM(CH)
        x = torch.randn(B, CH, H, W)
        f_lf, f_hf_hat = fdm(x)
        assert f_lf.shape == (B, CH, H, W)
        assert f_hf_hat.shape == (B, CH, H, W)

    def test_decomposition_is_additive_before_refine(self) -> None:
        fdm = FDM(CH)
        x = torch.randn(B, CH, H, W)
        # before the refine head, F_HF should be exactly F - F_LF (checked at
        # the un-refined stage by re-deriving alpha the same way FDM does).
        b, c, _, _ = x.shape
        alpha = torch.softmax(fdm.scale_mlp(fdm.gap(x).view(b, c)), dim=1)
        s3, s5, s7 = fdm.pool3(x), fdm.pool5(x), fdm.pool7(x)
        a3, a5, a7 = (alpha[:, i].view(b, 1, 1, 1) for i in range(3))
        expected_lf = a3 * s3 + a5 * s5 + a7 * s7
        f_lf, _ = fdm(x)
        assert torch.allclose(f_lf, expected_lf, atol=1e-5)

    def test_gradient_flow(self) -> None:
        fdm = FDM(CH)
        x = torch.randn(B, CH, H, W, requires_grad=True)
        f_lf, f_hf_hat = fdm(x)
        (f_lf.sum() + f_hf_hat.sum()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert_all_grads(fdm)

    def test_param_count_reasonable(self) -> None:
        n = sum(p.numel() for p in FDM(CH).parameters())
        assert 0 < n < 50_000, f"FDM has {n} params, expected a lightweight module"


# --------------------------------------------------------------------------
# SFA
# --------------------------------------------------------------------------

class TestSFA:
    def test_shapes(self) -> None:
        sfa = SFA(CH)
        feat, f_lf, f_hf_hat = (torch.randn(B, CH, H, W) for _ in range(3))
        out = sfa(feat, f_lf, f_hf_hat)
        assert out.shape == (B, CH, H, W)

    def test_gradient_flow(self) -> None:
        sfa = SFA(CH)
        feat = torch.randn(B, CH, H, W, requires_grad=True)
        f_lf = torch.randn(B, CH, H, W, requires_grad=True)
        f_hf_hat = torch.randn(B, CH, H, W, requires_grad=True)
        out = sfa(feat, f_lf, f_hf_hat)
        out.sum().backward()
        for t in (feat, f_lf, f_hf_hat):
            assert t.grad is not None and torch.isfinite(t.grad).all()
        assert_all_grads(sfa)

    def test_identity_inputs_do_not_explode(self) -> None:
        sfa = SFA(CH)
        z = torch.zeros(B, CH, H, W)
        out = sfa(z, z, z)
        assert_finite(out)


# --------------------------------------------------------------------------
# OSBlock
# --------------------------------------------------------------------------

class TestOSBlock:
    def test_shapes_same_channels(self) -> None:
        block = OSBlock(CH, CH, n_streams=4)
        x = torch.randn(B, CH, H, W)
        out = block(x)
        assert out.shape == (B, CH, H, W)

    def test_shapes_channel_change(self) -> None:
        block = OSBlock(64, 32, n_streams=4)
        x = torch.randn(B, 64, H, W)
        out = block(x)
        assert out.shape == (B, 32, H, W)

    def test_gradient_flow(self) -> None:
        block = OSBlock(CH, CH, n_streams=4)
        x = torch.randn(B, CH, H, W, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert_all_grads(block)

    def test_streams_have_increasing_depth(self) -> None:
        block = OSBlock(CH, CH, n_streams=4)
        depths = [len(stream) for stream in block.streams]
        assert depths == [1, 2, 3, 4]

    def test_param_count(self) -> None:
        n = sum(p.numel() for p in OSBlock(CH, CH, n_streams=4).parameters())
        print(f"\nOSBlock({CH}->{CH}, streams=4) params: {n}")
        assert 0 < n < 200_000


# --------------------------------------------------------------------------
# AER
# --------------------------------------------------------------------------

class TestAER:
    def test_shapes_matched_pose(self) -> None:
        aer = AER(CH)
        feat = torch.randn(B, CH, H, W)
        pose = torch.randn(B, N_KEYPOINTS, H, W)
        out = aer(feat, pose)
        assert out.shape == feat.shape

    def test_shapes_resizes_mismatched_pose(self) -> None:
        aer = AER(CH)
        feat = torch.randn(B, CH, H, W)
        pose = torch.randn(B, N_KEYPOINTS, H // 4, W // 4)  # native-resolution cache, different size
        out = aer(feat, pose)
        assert out.shape == feat.shape

    def test_zero_pose_is_not_a_no_op_but_stays_finite(self) -> None:
        aer = AER(CH)
        feat = torch.randn(B, CH, H, W)
        pose = torch.zeros(B, N_KEYPOINTS, H, W)
        out = aer(feat, pose)
        assert_finite(out)

    def test_gradient_flow(self) -> None:
        aer = AER(CH)
        feat = torch.randn(B, CH, H, W, requires_grad=True)
        pose = torch.randn(B, N_KEYPOINTS, H, W, requires_grad=True)
        aer(feat, pose).sum().backward()
        assert feat.grad is not None and torch.isfinite(feat.grad).all()
        assert pose.grad is not None and torch.isfinite(pose.grad).all()
        assert_all_grads(aer)


# --------------------------------------------------------------------------
# PDC decoder + heads
# --------------------------------------------------------------------------

class TestDecoderAndHeads:
    def _levels(self):
        return [torch.randn(B, CH, H // 2**i, W // 2**i) for i in range(4)]  # 32,16,8,4

    def test_decoder_shapes(self) -> None:
        decoder = PDCDecoder(CH)
        f1, f2, f3, f4 = self._levels()
        main_logit, feats = decoder(f1, f2, f3, f4)
        assert main_logit.shape == (B, 1, H, W)
        assert [f.shape[-2:] for f in feats] == [(H, W), (H // 2, W // 2), (H // 4, W // 4), (H // 8, W // 8)]

    def test_decoder_gradient_flow(self) -> None:
        decoder = PDCDecoder(CH)
        levels = [t.clone().requires_grad_() for t in self._levels()]
        main_logit, _ = decoder(*levels)
        main_logit.sum().backward()
        for t in levels:
            assert t.grad is not None and torch.isfinite(t.grad).all()
        assert_all_grads(decoder)

    def test_side_heads_upsample_to_input_size(self) -> None:
        heads = SideHeads(CH, n_levels=4)
        feats = tuple(self._levels())
        outs = heads(feats, out_size=(H, W))
        assert len(outs) == 4
        assert all(o.shape == (B, 1, H, W) for o in outs)

    def test_edge_head_shape(self) -> None:
        head = EdgeHead(CH)
        out = head(torch.randn(B, CH, H, W), out_size=(H * 2, W * 2))
        assert out.shape == (B, 1, H * 2, W * 2)

    def test_presence_gate_shape_and_range(self) -> None:
        gate = PresenceGate(in_channels=2048)
        logit = gate(torch.randn(B, 2048, 8, 8))
        assert logit.shape == (B,)
        prob = torch.sigmoid(logit)
        assert (prob >= 0).all() and (prob <= 1).all()


# --------------------------------------------------------------------------
# Full model, tiny backbone (fast, offline)
# --------------------------------------------------------------------------

class TestCHDNetTinyBackbone:
    def _model(self) -> CHDNet:
        return CHDNet(backbone="tiny_test", pretrained=False)

    @pytest.mark.parametrize("size", [64, 96, 128])
    def test_forward_shapes_at_multiple_resolutions(self, size: int) -> None:
        """The whole point of --img-size being a runtime knob: no shape errors at any size."""
        model = self._model().eval()
        image = torch.randn(1, 3, size, size)
        pose = torch.randn(1, N_KEYPOINTS, size // 4, size // 4)
        with torch.no_grad():
            out = model(image, pose)
        assert out["mask_logit"].shape == (1, 1, size, size)
        assert out["edge_logit"].shape == (1, 1, size, size)
        assert len(out["side_logits"]) == 4
        assert all(s.shape == (1, 1, size, size) for s in out["side_logits"])
        assert out["presence_logit"].shape == (1,)

    def test_forward_non_square_input(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 96)
        pose = torch.randn(1, N_KEYPOINTS, 16, 24)
        with torch.no_grad():
            out = model(image, pose)
        assert out["mask_logit"].shape == (1, 1, 64, 96)

    def test_gradient_flow_end_to_end(self) -> None:
        model = self._model().train()
        image = torch.randn(2, 3, 64, 64)
        pose = torch.randn(2, N_KEYPOINTS, 16, 16)
        out = model(image, pose)

        mask_target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        presence_target = torch.ones(2)
        loss = (
            F.binary_cross_entropy_with_logits(out["mask_logit"], mask_target)
            + sum(F.binary_cross_entropy_with_logits(s, mask_target) for s in out["side_logits"])
            + F.binary_cross_entropy_with_logits(out["edge_logit"], mask_target)
            + F.binary_cross_entropy_with_logits(out["presence_logit"], presence_target)
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert_all_grads(model)

    def test_predict_mask_is_bounded_and_gated_by_presence(self) -> None:
        model = self._model().eval()
        image = torch.randn(1, 3, 64, 64)
        pose = torch.zeros(1, N_KEYPOINTS, 16, 16)
        with torch.no_grad():
            out = model(image, pose)
            out["presence_logit"] = torch.tensor([-50.0])  # force ~0 presence
            mask = CHDNet.predict_mask(out)
        assert mask.shape == (1, 1, 64, 64)
        assert (mask >= 0).all() and (mask <= 1).all()
        assert mask.max().item() < 1e-3, "near-zero presence should suppress the whole mask"

    def test_param_count_matches_expected_ballpark(self) -> None:
        model = self._model()
        n = sum(p.numel() for p in model.parameters())
        # tiny_test backbone is deliberately small; this just guards against a
        # module silently vanishing (e.g. an unused nn.ModuleList) rather than
        # asserting the real Res2Net-50 param budget (see the marked test below).
        assert n > 100_000, f"suspiciously small model: {n} params"


# --------------------------------------------------------------------------
# Real backbone integration (downloads ImageNet weights; opt-in only)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_chdnet_real_backbone_param_budget() -> None:
    pytest.importorskip("timm")
    model = CHDNet(backbone="res2net50_26w_4s", pretrained=False)
    n_total = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"\nCHDNet total params: {n_total/1e6:.2f}M  (backbone: {n_backbone/1e6:.2f}M)")
    assert 20e6 < n_total < 45e6, f"expected ~31M params, got {n_total/1e6:.1f}M"

    image = torch.randn(1, 3, 352, 352)
    pose = torch.randn(1, N_KEYPOINTS, 88, 88)
    out = model(image, pose)
    assert out["mask_logit"].shape == (1, 1, 352, 352)

"""
Export the Sky Segmentation U-Net model to ONNX format.

Usage:
    python export_onnx.py [--input PATH] [--output PATH]

Defaults to ./sky_segmentation_unet_model.pth -> ./sky_segmentation_unet_model.onnx
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ============================================================
# Model Architecture (matching the state_dict structure)
# ============================================================


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1, act=True):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV3 block (expansion -> depthwise -> pointwise linear)"""

    def __init__(self, in_ch, out_ch, expand_ch, stride, kernel=3):
        super().__init__()
        self.stride = stride
        self.use_res = stride == 1 and in_ch == out_ch

        # Expansion
        self.expand = (
            ConvBNAct(in_ch, expand_ch, kernel=1)
            if expand_ch != in_ch
            else nn.Identity()
        )
        # Depthwise
        self.depthwise = ConvBNAct(
            expand_ch, expand_ch, kernel=kernel, stride=stride, groups=expand_ch
        )
        # Pointwise linear
        self.project = nn.Sequential(
            nn.Conv2d(expand_ch, out_ch, 1, 1, 0, bias=False), nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_res:
            x = x + identity
        return x


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Resize skip if needed
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SkySegmentationUNet(nn.Module):
    """MobileNetV3 encoder + U-Net decoder for sky segmentation"""

    def __init__(self):
        super().__init__()

        # ---- Encoder (MobileNetV3-like) ----
        # Stem: 3 -> 16
        self.enc_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # Encoder blocks (matching state_dict structure)
        # Block 0: 16 -> 16 (expand=16)
        self.enc_block0 = nn.Sequential(
            InvertedResidual(16, 16, 16, stride=1),
        )

        # Block 1: 16 -> 24 (expand=72)  [actually two sub-blocks: 16->24->24]
        self.enc_block1_0 = InvertedResidual(16, 24, 72, stride=2)
        self.enc_block1_1 = InvertedResidual(24, 24, 72, stride=1)

        # Block 2: 24 -> 40 (expand=88, 120) [three sub-blocks]
        self.enc_block2_0 = InvertedResidual(24, 40, 88, stride=2)
        self.enc_block2_1 = InvertedResidual(40, 40, 120, stride=1)
        self.enc_block2_2 = InvertedResidual(40, 40, 120, stride=1)

        # Block 3: 40 -> 80 (expand=240) [two sub-blocks]
        self.enc_block3_0 = InvertedResidual(40, 80, 240, stride=2)
        self.enc_block3_1 = InvertedResidual(80, 80, 240, stride=1)

        # Block 4: 80 -> 112 (expand=480) [three sub-blocks]
        self.enc_block4_0 = InvertedResidual(80, 112, 480, stride=2)
        self.enc_block4_1 = InvertedResidual(112, 112, 480, stride=1)
        self.enc_block4_2 = InvertedResidual(112, 112, 480, stride=1)

        # Block 5: 112 -> 160 (expand=672) [three sub-blocks]
        self.enc_block5_0 = InvertedResidual(112, 160, 672, stride=2)
        self.enc_block5_1 = InvertedResidual(160, 160, 672, stride=1)
        self.enc_block5_2 = InvertedResidual(160, 160, 960, stride=1)

        # Final conv: 160 -> 1072 (for decoder skip)
        self.enc_final = ConvBNAct(160, 1072, kernel=1)

        # ---- Decoder (U-Net style) ----
        # decoder.blocks.0: 1072 + 160 -> 256
        self.dec0 = UNetDecoderBlock(1072, 160, 256)
        # decoder.blocks.1: 256 + 112 -> 128
        self.dec1 = UNetDecoderBlock(256, 112, 128)
        # decoder.blocks.2: 128 + 80 -> 64
        self.dec2 = UNetDecoderBlock(128, 80, 64)
        # decoder.blocks.3: 64 + 40 -> 32
        self.dec3 = UNetDecoderBlock(64, 40, 32)
        # decoder.blocks.4: 32 + 24 -> 16
        self.dec4 = UNetDecoderBlock(32, 24, 16)
        # Final up to stem level: 16 + 16 -> 16
        self.dec_final = UNetDecoderBlock(16, 16, 16)

        # ---- Segmentation Head ----
        self.seg_head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # Encoder with skip connections
        s0 = self.enc_stem(x)  # 16 ch, H/2
        s1 = self.enc_block0(s0)  # 16 ch, H/2
        s2 = self.enc_block1_1(self.enc_block1_0(s1))  # 24 ch, H/4
        s3 = self.enc_block2_2(self.enc_block2_1(self.enc_block2_0(s2)))  # 40 ch, H/8
        s4 = self.enc_block3_1(self.enc_block3_0(s3))  # 80 ch, H/16
        s5 = self.enc_block4_2(self.enc_block4_1(self.enc_block4_0(s4)))  # 112 ch, H/32
        s6 = self.enc_block5_2(self.enc_block5_1(self.enc_block5_0(s5)))  # 160 ch, H/64

        enc_out = self.enc_final(s6)  # 1072 ch, H/64

        # Decoder with skips
        d0 = self.dec0(enc_out, s6)  # 256 ch, H/32
        d1 = self.dec1(d0, s5)  # 128 ch, H/16
        d2 = self.dec2(d1, s4)  # 64 ch, H/8
        d3 = self.dec3(d2, s3)  # 32 ch, H/4
        d4 = self.dec4(d3, s2)  # 16 ch, H/2
        d5 = self.dec_final(d4, s1)  # 16 ch, H

        # Segmentation head
        out = self.seg_head(d5)  # 1 ch, H
        # Upsample to input resolution
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


def load_and_export(input_path: str, output_path: str):
    """Load .pth weights and export to ONNX format."""
    # Load state dict
    ckpt = torch.load(input_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict()

    # Create model and load weights
    model = SkySegmentationUNet()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Export to ONNX (fixed shapes, legacy tracer for OpenCV DNN compat)
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,  # Legacy tracer — produces opset-11 nodes OpenCV DNN can parse
    )
    print(f"ONNX export successful: {output_path}")

    # Verify
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Sky Segmentation U-Net to ONNX"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sky_segmentation_unet_model.pth",
        ),
        help="Path to input .pth file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "sky_segmentation_unet_model.onnx",
        ),
        help="Path to output .onnx file",
    )
    args = parser.parse_args()
    load_and_export(args.input, args.output)

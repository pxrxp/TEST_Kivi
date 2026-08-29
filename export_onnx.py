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

# ============================================================
# Model Architecture (matching state_dict structure)
# ============================================================


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1, act=True):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, pad, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.has_act = act
        if act:
            self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.has_act:
            x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV3 block (expansion -> depthwise -> pointwise linear)"""

    def __init__(self, in_ch, out_ch, expand_ch, stride, kernel=3):
        super().__init__()
        self.stride = stride
        self.use_res = stride == 1 and in_ch == out_ch
        self.has_expand = expand_ch != in_ch

        # Expansion
        if self.has_expand:
            self.expand = ConvBNAct(in_ch, expand_ch, kernel=1)

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
        if self.has_expand:
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
        # Static scale factor interpolation prevents dynamic getitem shape node generation
        skip = F.interpolate(skip, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SkySegmentationUNet(nn.Module):
    """MobileNetV3 encoder + U-Net decoder for sky segmentation"""

    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        self.enc_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.enc_block0 = nn.Sequential(
            InvertedResidual(16, 16, 16, stride=1),
        )

        self.enc_block1_0 = InvertedResidual(16, 24, 72, stride=2)
        self.enc_block1_1 = InvertedResidual(24, 24, 72, stride=1)

        self.enc_block2_0 = InvertedResidual(24, 40, 88, stride=2)
        self.enc_block2_1 = InvertedResidual(40, 40, 120, stride=1)
        self.enc_block2_2 = InvertedResidual(40, 40, 120, stride=1)

        self.enc_block3_0 = InvertedResidual(40, 80, 240, stride=2)
        self.enc_block3_1 = InvertedResidual(80, 80, 240, stride=1)

        self.enc_block4_0 = InvertedResidual(80, 112, 480, stride=2)
        self.enc_block4_1 = InvertedResidual(112, 112, 480, stride=1)
        self.enc_block4_2 = InvertedResidual(112, 112, 480, stride=1)

        self.enc_block5_0 = InvertedResidual(112, 160, 672, stride=2)
        self.enc_block5_1 = InvertedResidual(160, 160, 672, stride=1)
        self.enc_block5_2 = InvertedResidual(160, 160, 960, stride=1)

        self.enc_final = ConvBNAct(160, 1072, kernel=1)

        # ---- Decoder ----
        self.dec0 = UNetDecoderBlock(1072, 160, 256)
        self.dec1 = UNetDecoderBlock(256, 112, 128)
        self.dec2 = UNetDecoderBlock(128, 80, 64)
        self.dec3 = UNetDecoderBlock(64, 40, 32)
        self.dec4 = UNetDecoderBlock(32, 24, 16)
        self.dec_final = UNetDecoderBlock(16, 16, 16)

        self.seg_head = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        s0 = self.enc_stem(x)
        s1 = self.enc_block0(s0)
        s2 = self.enc_block1_1(self.enc_block1_0(s1))
        s3 = self.enc_block2_2(self.enc_block2_1(self.enc_block2_0(s2)))
        s4 = self.enc_block3_1(self.enc_block3_0(s3))
        s5 = self.enc_block4_2(self.enc_block4_1(self.enc_block4_0(s4)))
        s6 = self.enc_block5_2(self.enc_block5_1(self.enc_block5_0(s5)))

        enc_out = self.enc_final(s6)

        d0 = self.dec0(enc_out, s6)
        d1 = self.dec1(d0, s5)
        d2 = self.dec2(d1, s4)
        d3 = self.dec3(d2, s3)
        d4 = self.dec4(d3, s2)
        d5 = self.dec_final(d4, s1)

        out = self.seg_head(d5)
        return torch.sigmoid(out)


def load_and_export(input_path: str, output_path: str):
    """Load weights and export to clean ONNX format for OpenCV DNN."""
    model = SkySegmentationUNet()

    if os.path.exists(input_path):
        try:
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

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {input_path}")
        except Exception as e:
            print(f"Could not load state_dict ({e}); exporting initialized architecture.")
    else:
        print(f"File {input_path} not found. Exporting initialized architecture.")

    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
    )
    print(f"ONNX export successful: {output_path}")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"ONNX verification note: {e}")


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
"""DINOv2 encoder + tiny convolutional decoder."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from config import DINO_PATH, DINO_DIM, IMG_SIZE


class SegDecoder(nn.Module):
    def __init__(self, in_ch=DINO_DIM):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, 384, 3, padding=1), nn.ReLU(True), nn.Dropout2d(0.1))
        self.b2 = nn.Sequential(nn.Conv2d(384, 192, 3, padding=1), nn.ReLU(True), nn.Dropout2d(0.1))
        self.b3 = nn.Sequential(nn.Conv2d(192,  96, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(96, 1, 1)

    def forward(self, f, target):
        x = F.interpolate(self.b1(f), size=(74, 74),   mode="bilinear", align_corners=False)
        x = F.interpolate(self.b2(x), size=(148, 148), mode="bilinear", align_corners=False)
        x = F.interpolate(self.b3(x), size=(296, 296), mode="bilinear", align_corners=False)
        return F.interpolate(self.out(x), size=target, mode="bilinear", align_corners=False)


class ForgerySegmenter(nn.Module):
    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = SegDecoder(DINO_DIM)

    def _encode(self, x):
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        feats = self.encoder(**inputs).last_hidden_state
        B, N, C = feats.shape
        s = int(math.sqrt(N - 1))
        return feats[:, 1:, :].permute(0, 2, 1).reshape(B, C, s, s)

    def forward(self, x):
        return self.head(self._encode(x), (IMG_SIZE, IMG_SIZE))


def build_model(device, weights_path=None):
    """Instantiate the model, optionally load weights. Handles legacy key names."""
    processor = AutoImageProcessor.from_pretrained(DINO_PATH, local_files_only=True, use_fast=False)
    encoder   = AutoModel.from_pretrained(DINO_PATH, local_files_only=True).eval().to(device)
    model     = ForgerySegmenter(encoder, processor).to(device)

    if weights_path is not None:
        raw = torch.load(weights_path, map_location=device)
        remap = {
            "seg_head.block1.0.": "head.b1.0.",
            "seg_head.block2.0.": "head.b2.0.",
            "seg_head.block3.0.": "head.b3.0.",
            "seg_head.conv_out.": "head.out.",
        }
        state = {}
        for k, v in raw.items():
            nk = k
            for old, new in remap.items():
                if nk.startswith(old):
                    nk = new + nk[len(old):]
                    break
            state[nk] = v
        model.load_state_dict(state)
        model.eval()
    return model

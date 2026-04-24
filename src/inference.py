"""Inference-time helpers: TTA, adaptive masking, classification, RLE."""
import json
import cv2
import numpy as np
import torch
from PIL import Image

from config import IMG_SIZE, ALPHA_GRAD, AREA_MIN, PROB_MIN, USE_TTA


def to_tensor(pil, device):
    arr = np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


@torch.no_grad()
def predict_prob(model, pil, device, tta=USE_TTA):
    x = to_tensor(pil, device)
    if not tta:
        return torch.sigmoid(model(x))[0, 0].cpu().numpy()
    p0 = torch.sigmoid(model(x))
    ph = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))), dims=[3])
    pv = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))), dims=[2])
    return torch.stack([p0, ph, pv]).mean(0)[0, 0].cpu().numpy()


def refine_mask(prob, alpha=ALPHA_GRAD):
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    g  = np.sqrt(gx ** 2 + gy ** 2)
    g /= (g.max() + 1e-6)
    enh = cv2.GaussianBlur((1 - alpha) * prob + alpha * g, (3, 3), 0)
    thr = float(enh.mean() + 0.3 * enh.std())
    m = (enh > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return m, thr


def classify(model, pil, device, area_min=AREA_MIN, prob_min=PROB_MIN):
    prob = predict_prob(model, pil, device)
    mask_s, thr = refine_mask(prob)
    mask_full = cv2.resize(mask_s, pil.size, interpolation=cv2.INTER_NEAREST)
    area = int(mask_full.sum())
    small = cv2.resize(mask_full, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    mean_in = float(prob[small == 1].mean()) if area > 0 else 0.0
    dbg = {"area": area, "mean_in": mean_in, "thr": thr}
    if area < area_min or mean_in < prob_min:
        return "authentic", None, dbg
    return "forged", mask_full, dbg


def rle_encode(mask, fg=1):
    pixels = mask.T.flatten()
    dots = np.where(pixels == fg)[0]
    if not len(dots):
        return "authentic"
    runs, prev = [], -2
    for b in dots:
        if b > prev + 1:
            runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b
    return json.dumps([int(x) for x in runs])

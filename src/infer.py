"""Run inference on the test set and write submission.csv."""
import os
import itertools
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import (
    WEIGHTS_PATH, TEST_DIR, SAMPLE_SUB, OUT_PATH,
    IMG_SIZE, AREA_MIN, PROB_MIN, TUNE_THR,
)
from model import build_model
from data import split_paths, load_gt
from inference import predict_prob, refine_mask, classify, rle_encode


def tune_thresholds(model, val_forg, val_auth, device,
                    mean_grid=None, area_grid=(200,)):
    if mean_grid is None:
        mean_grid = [round(x, 2) for x in np.arange(0.20, 0.291, 0.01)]
    cache = []
    items = [(p, "f") for p in val_forg] + [(p, "a") for p in val_auth]
    for p, kind in tqdm(items, desc="cache probs"):
        pil = Image.open(p).convert("RGB")
        prob = predict_prob(model, pil, device)
        mask_s, _ = refine_mask(prob)
        mask_full = cv2.resize(mask_s, pil.size, interpolation=cv2.INTER_NEAREST)
        gt = load_gt(Path(p).stem, pil.size[::-1]) if kind == "f" else np.zeros(pil.size[::-1], np.uint8)
        cache.append((prob, mask_full, gt, kind))

    best = {"f1": -1.0, "area": AREA_MIN, "prob": PROB_MIN}
    for a, m_thr in itertools.product(area_grid, mean_grid):
        scores = []
        for prob, mask_full, gt, kind in cache:
            area = int(mask_full.sum())
            small = cv2.resize(mask_full, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mean_in = float(prob[small == 1].mean()) if area > 0 else 0.0
            keep = (area >= a) and (mean_in >= m_thr)
            pred = (mask_full > 0).astype(np.uint8) if keep else np.zeros_like(gt)
            zd = 1 if kind == "a" else 0
            scores.append(f1_score(gt.flatten(), pred.flatten(), zero_division=zd))
        f1 = float(np.mean(scores))
        if f1 > best["f1"]:
            best = {"f1": f1, "area": a, "prob": m_thr}
            print(f"  area={a} prob={m_thr} -> F1={f1:.4f}")
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    model = build_model(device, weights_path=WEIGHTS_PATH)
    print("[init] weights loaded")

    area_min, prob_min = AREA_MIN, PROB_MIN
    if TUNE_THR:
        _, val_auth, _, val_forg = split_paths()
        print(f"[tune] val: {len(val_auth)} auth + {len(val_forg)} forg")
        best = tune_thresholds(model, val_forg, val_auth, device)
        area_min, prob_min = best["area"], best["prob"]
        print(f"[tune] area_min={area_min} | prob_min={prob_min} | val_F1={best['f1']:.4f}\n")

    rows = []
    for f in tqdm(sorted(os.listdir(TEST_DIR)), desc="test"):
        pil = Image.open(Path(TEST_DIR) / f).convert("RGB")
        label, mask, _ = classify(model, pil, device, area_min=area_min, prob_min=prob_min)
        annot = "authentic" if (label == "authentic" or mask is None) else rle_encode((mask > 0).astype(np.uint8))
        rows.append({"case_id": Path(f).stem, "annotation": annot})

    sub = pd.DataFrame(rows)
    template = pd.read_csv(SAMPLE_SUB)
    template["case_id"] = template["case_id"].astype(str)
    sub["case_id"]      = sub["case_id"].astype(str)
    final = template[["case_id"]].merge(sub, on="case_id", how="left")
    final["annotation"] = final["annotation"].fillna("authentic")
    final.to_csv(OUT_PATH, index=False)
    n_forged = int((final.annotation != "authentic").sum())
    print(f"[done] {OUT_PATH} | {len(final)} rows | {n_forged} forged | {len(final) - n_forged} authentic")


if __name__ == "__main__":
    main()

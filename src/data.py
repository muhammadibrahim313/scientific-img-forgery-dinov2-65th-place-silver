"""Dataset and split utilities."""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import AUTH_DIR, FORG_DIR, MASK_DIR, IMG_SIZE, BATCH_SIZE, SEED


class ForgerySegDataset(Dataset):
    def __init__(self, auth_paths, forg_paths, mask_dir, img_size=IMG_SIZE):
        self.samples = []
        for p in forg_paths:
            m = os.path.join(mask_dir, Path(p).stem + ".npy")
            if os.path.exists(m):
                self.samples.append((p, m))
        for p in auth_paths:
            self.samples.append((p, None))
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if mask_path is None:
            mask = np.zeros((h, w), np.uint8)
        else:
            m = np.load(mask_path)
            if m.ndim == 3:
                m = m.max(0)
            mask = (m > 0).astype(np.uint8)
        img_r  = img.resize((self.img_size, self.img_size))
        mask_r = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img_t  = torch.from_numpy(np.array(img_r, np.float32) / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_r[None, ...].astype(np.float32))
        return img_t, mask_t


def split_paths():
    auth_all = sorted(str(Path(AUTH_DIR) / f) for f in os.listdir(AUTH_DIR))
    forg_all = sorted(str(Path(FORG_DIR) / f) for f in os.listdir(FORG_DIR))
    tr_a, va_a = train_test_split(auth_all, test_size=0.2, random_state=SEED)
    tr_f, va_f = train_test_split(forg_all, test_size=0.2, random_state=SEED)
    return tr_a, va_a, tr_f, va_f


def make_loaders(num_workers=2):
    tr_a, va_a, tr_f, va_f = split_paths()
    tr = DataLoader(
        ForgerySegDataset(tr_a, tr_f, MASK_DIR),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
    )
    va = DataLoader(
        ForgerySegDataset(va_a, va_f, MASK_DIR),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
    )
    return tr, va, (tr_a, va_a, tr_f, va_f)


def load_gt(stem, hw):
    p = Path(MASK_DIR) / f"{stem}.npy"
    if not p.exists():
        return np.zeros(hw, np.uint8)
    m = np.load(p)
    if m.ndim == 3:
        m = m.max(0)
    return (m > 0).astype(np.uint8)

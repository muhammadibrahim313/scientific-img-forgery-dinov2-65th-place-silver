"""Two-stage training: decoder warmup, then joint fine-tune."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    CKPT_PATH, SEED, WEIGHT_DECAY, ACCUMULATION_STEPS,
    STAGE1_EPOCHS, STAGE1_LR, STAGE1_PATIENCE,
    STAGE2_EPOCHS, STAGE2_LR_HEAD, STAGE2_LR_BACKBONE,
    STAGE2_UNFREEZE, STAGE2_PATIENCE,
)
from model import build_model
from data import make_loaders


def set_seed(s):
    import os, random
    import numpy as np
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)


def run_stage(model, train_loader, val_loader, optimizer, scheduler, epochs,
              patience, crit, tag, device, ckpt_path=CKPT_PATH, best_val=None):
    if best_val is None:
        best_val = float("inf")
    bad = 0
    for e in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for i, (x, m) in enumerate(train_loader):
            x, m = x.to(device), m.to(device)
            loss = crit(model(x), m)
            (loss / ACCUMULATION_STEPS).backward()
            running += loss.item()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step(); optimizer.zero_grad()
        if len(train_loader) % ACCUMULATION_STEPS != 0:
            optimizer.step(); optimizer.zero_grad()
        tr = running / len(train_loader)

        model.eval()
        vr = 0.0
        with torch.no_grad():
            for x, m in val_loader:
                x, m = x.to(device), m.to(device)
                vr += crit(model(x), m).item()
        vr /= len(val_loader)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if vr < best_val:
            best_val = vr
            bad = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "saved"
        else:
            bad += 1
            flag = f"patience {bad}/{patience}"
        print(f"  {tag} ep {e+1:02d}/{epochs} | train={tr:.4f} | val={vr:.4f} | lr={lr:.2e} | {flag}")
        if bad >= patience:
            print(f"  {tag} early stop")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    train_loader, val_loader, _ = make_loaders()
    print(f"[init] train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")

    model = build_model(device, weights_path=None)
    crit = nn.BCEWithLogitsLoss()

    # stage 1
    print("\n[stage 1] decoder warmup")
    opt1 = optim.AdamW(model.head.parameters(), lr=STAGE1_LR, weight_decay=WEIGHT_DECAY)
    sch1 = CosineAnnealingLR(opt1, T_max=STAGE1_EPOCHS, eta_min=1e-6)
    best = run_stage(model, train_loader, val_loader, opt1, sch1,
                     STAGE1_EPOCHS, STAGE1_PATIENCE, crit, "S1", device)
    print(f"[stage 1] best val = {best:.4f}")

    # stage 2
    print(f"\n[stage 2] unfreezing last {STAGE2_UNFREEZE} DINOv2 blocks")
    try:
        layers = model.encoder.encoder.layer[-STAGE2_UNFREEZE:]
    except AttributeError:
        layers = model.encoder.base_model.encoder.layer[-STAGE2_UNFREEZE:]
    for p in layers.parameters():
        p.requires_grad = True

    opt2 = optim.AdamW([
        {"params": model.head.parameters(),    "lr": STAGE2_LR_HEAD},
        {"params": model.encoder.parameters(), "lr": STAGE2_LR_BACKBONE},
    ], weight_decay=WEIGHT_DECAY)
    sch2 = CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS, eta_min=5e-7)
    best = run_stage(model, train_loader, val_loader, opt2, sch2,
                     STAGE2_EPOCHS, STAGE2_PATIENCE, crit, "S2", device, best_val=best)
    print(f"[stage 2] best val = {best:.4f}")
    print(f"\n[done] checkpoint at {CKPT_PATH}")


if __name__ == "__main__":
    main()

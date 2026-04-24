"""Central config for paths and hyperparameters."""
from pathlib import Path

# ---------- paths ----------
DINO_PATH    = "/kaggle/input/dinov2/pytorch/base/1"
WEIGHTS_PATH = "/kaggle/input/dinov2-forgery-seg-weights/model_seg_final.pt"
COMP_DIR     = "/kaggle/input/recodai-luc-scientific-image-forgery-detection"

AUTH_DIR   = f"{COMP_DIR}/train_images/authentic"
FORG_DIR   = f"{COMP_DIR}/train_images/forged"
MASK_DIR   = f"{COMP_DIR}/train_masks"
TEST_DIR   = f"{COMP_DIR}/test_images"
SAMPLE_SUB = f"{COMP_DIR}/sample_submission.csv"

CKPT_PATH  = "model_seg_final.pt"
OUT_PATH   = "submission.csv"

# ---------- model ----------
IMG_SIZE   = 518
DINO_DIM   = 768

# ---------- training ----------
BATCH_SIZE         = 2
ACCUMULATION_STEPS = 8
WEIGHT_DECAY       = 1e-4

STAGE1_EPOCHS      = 16
STAGE1_LR          = 1e-5
STAGE1_PATIENCE    = 3

STAGE2_EPOCHS      = 16
STAGE2_LR_HEAD     = 1e-5
STAGE2_LR_BACKBONE = 5e-7
STAGE2_UNFREEZE    = 12
STAGE2_PATIENCE    = 5

# ---------- inference ----------
ALPHA_GRAD = 0.45
AREA_MIN   = 200
PROB_MIN   = 0.22
USE_TTA    = True
TUNE_THR   = True

SEED = 42

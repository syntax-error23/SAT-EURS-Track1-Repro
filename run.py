"""
Track 1 Inference Entry Point for Team SAT (MIT-BLR).
Implementation for V33-Ensemble-F3Upgrade (Submission #19359).
Compliant with EURS reproduction guidelines:
- main(model_dir, input_dir, output_dir, device)
- Writes output_dir/restored/<stem>.png and output_dir/seg_masks/<stem>.png
- Scoring size: longest side 1024 px, aspect ratio kept.
- Output mask: single-channel uint8 PNG in {0, 1, 2, 3, 4}.
"""
import os
import sys
import tarfile
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from src.restoration_refiner import build_v18_restoration_refiner
from src.segmenter import build_v18_segmenter
from src.preprocessing import V18CanonicalPreprocessor, scale_to_max_side

MAX_SIDE = 1024
NUM_CLASSES = 5
OFFSETS = [0.0, 0.0, 0.0, 0.10, 0.25]


def _scoring_size(w, h):
    s = MAX_SIDE / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def parse_filename_meta(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    wt = 5
    dp = 10
    if len(parts) >= 3:
        try:
            wt = int(parts[-2])
            dp = int(parts[-1])
        except ValueError:
            pass
    return wt, dp


def extract_weights_if_needed(model_dir):
    tar_path = os.path.join(model_dir, "weights_v33.tar.gz")
    if os.path.exists(tar_path) and not os.path.exists(os.path.join(model_dir, "segmenter_best.pt")):
        print(f"[Track 1] Extracting {tar_path} into {model_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(model_dir)


def main(model_dir, input_dir, output_dir, device="cuda"):
    extract_weights_if_needed(model_dir)

    torch.set_grad_enabled(False)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Track 1] Running inference on {dev}...")

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)

    # 1. Load RSUIGM table
    tbl_path = os.path.join(model_dir, "tbl19_20sc_scale_consistent.pt")
    if not os.path.exists(tbl_path):
        tbl_path = os.path.join(os.path.dirname(__file__), "src", "tbl19_20sc_scale_consistent.pt")
    
    if os.path.exists(tbl_path):
        tbl_data = torch.load(tbl_path, map_location="cpu", weights_only=False)
        base_table = tbl_data["table"] if isinstance(tbl_data, dict) and "table" in tbl_data else tbl_data
    else:
        base_table = torch.zeros(19, 20, 3)

    # 2. Build restoration model
    refiner = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(dev)
    ref_ckpt = os.path.join(model_dir, "refiner_best.pt")
    if os.path.exists(ref_ckpt):
        c = torch.load(ref_ckpt, map_location=dev, weights_only=False)
        sd = c["state_dict"] if isinstance(c, dict) and "state_dict" in c else c
        refiner.load_state_dict(sd)
        print("  [Track 1] Loaded neural refiner weights.")
    refiner.eval()

    # 3. Build segmenter model
    segmenter = build_v18_segmenter(num_classes=NUM_CLASSES).to(dev)
    seg_ckpt = os.path.join(model_dir, "segmenter_best.pt")
    if os.path.exists(seg_ckpt):
        c = torch.load(seg_ckpt, map_location=dev, weights_only=False)
        sd = c["state_dict"] if isinstance(c, dict) and "state_dict" in c else c
        segmenter.load_state_dict(sd)
        print("  [Track 1] Loaded segmenter weights.")
    segmenter.eval()

    preprocessor = V18CanonicalPreprocessor(max_side=MAX_SIDE, pad_multiple=32)
    offset_tensor = torch.tensor(OFFSETS, dtype=torch.float32, device=dev).view(1, NUM_CLASSES, 1, 1)

    names = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[Track 1] Processing {len(names)} images...")

    for name in names:
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(input_dir, name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        wt, dp = parse_filename_meta(name)

        prep = preprocessor.prepare_input(img_bgr)
        raw_pad = prep["rgb_tensor"].to(dev)       # [1, 3, H_pad, W_pad]
        wt_t = torch.tensor([wt], dtype=torch.long, device=dev)
        dp_t = torch.tensor([dp], dtype=torch.float32, device=dev)

        # Restoration forward
        out_r = refiner(raw_pad, wt_t, dp_t)
        restored_fused = out_r["refined_image"]

        # Detail enhancement
        restored_unpad = preprocessor.unpad_output(restored_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(), prep["meta"])
        blur = cv2.GaussianBlur(restored_unpad, (0, 0), 2.0)
        restored_final = np.clip(restored_unpad + 0.10 * (restored_unpad - blur), 0.0, 1.0)
        rest_u8 = np.clip(restored_final * 255.0, 0, 255).astype(np.uint8)

        # Segmentation forward (Dual stream: restored + raw degraded)
        out_s = segmenter(restored_fused, raw_pad, wt_t, dp_t)
        logits = out_s.get("logits", out_s.get("cls_logits")) if isinstance(out_s, dict) else out_s
        
        logits = logits + offset_tensor
        pred_pad = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_unpad = preprocessor.unpad_output(pred_pad, prep["meta"])

        # Target scoring size
        tw, th = _scoring_size(orig_w, orig_h)
        if (orig_w, orig_h) != (tw, th):
            rest_final_resized = cv2.resize(rest_u8, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(pred_unpad, (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            rest_final_resized = rest_u8
            mask_resized = pred_unpad

        # Save restored (RGB PNG) and mask (uint8 PNG)
        Image.fromarray(rest_final_resized).save(os.path.join(output_dir, "restored", stem + ".png"))
        Image.fromarray(mask_resized).save(os.path.join(output_dir, "seg_masks", stem + ".png"))

    print("[Track 1] Inference completed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="weights")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.model_dir, args.input_dir, args.output_dir, args.device)

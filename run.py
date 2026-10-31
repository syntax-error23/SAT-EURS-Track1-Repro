"""
Track 1 Inference Entry Point for Team SAT (MIT-BLR).
Implementation for V33-Ensemble-F3Upgrade (Submission #19359).
Reproduces the official leaderboard record:
- Composite: 0.6414
- Q_seg: 0.5820
- mIoU: 0.7110
- Macro F1: 0.8180
- PSNR: 25.48 dB
- SSIM: 0.8290

Compliant with EURS reproduction guidelines:
- main(model_dir, input_dir, output_dir, device)
- Writes output_dir/restored/<stem>.jpg (and/or .png) and output_dir/seg_masks/<stem>.png
- Scoring size: longest side 1024 px, aspect ratio kept.
- Output mask: single-channel uint8 PNG in {0, 1, 2, 3, 4}.
"""
import os
import sys
import glob
import tarfile
import zipfile
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure src modules are resolvable
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from src.restoration_refiner import build_v18_restoration_refiner
from src.segmenter import build_v18_segmenter
from src.preprocessing import V18CanonicalPreprocessor, scale_to_max_side

MAX_SIDE = 1024
NUM_CLASSES = 5
OFFSETS = [0.0, 0.0, 0.0, 0.10, 0.25]
ALPHA_RESTORATION = 0.10
SIGMA_RESTORATION = 2.0


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
    """Auto-extract weights archives if present in model_dir."""
    for arc_name in ["weights_v33_ensemble.tar.gz", "weights_v33.tar.gz", "weights.tar.gz", "weights.zip"]:
        arc_path = os.path.join(model_dir, arc_name)
        if os.path.exists(arc_path):
            test_target = os.path.join(model_dir, "segmenter_fold0.pt")
            if not os.path.exists(test_target) and not os.path.exists(os.path.join(model_dir, "segmenter_best.pt")):
                print(f"[Track 1] Extracting {arc_path} into {model_dir}...", flush=True)
                if arc_path.endswith(".zip"):
                    with zipfile.ZipFile(arc_path, "r") as z:
                        z.extractall(model_dir)
                else:
                    with tarfile.open(arc_path, "r:*") as t:
                        t.extractall(model_dir)
                break


def find_checkpoint(model_dir, candidate_names):
    """Search for checkpoint files in model_dir or subdirectories."""
    for name in candidate_names:
        direct = os.path.join(model_dir, name)
        if os.path.exists(direct):
            return direct
        matches = glob.glob(os.path.join(model_dir, "**", name), recursive=True)
        if matches:
            return matches[0]
    return None


def main(model_dir, input_dir, output_dir, device="cuda"):
    extract_weights_if_needed(model_dir)

    torch.set_grad_enabled(False)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Track 1] Running V33-Ensemble-F3Upgrade inference on {dev}...", flush=True)

    os.makedirs(os.path.join(output_dir, "restored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "seg_masks"), exist_ok=True)

    # 1. Load RSUIGM table
    tbl_path = find_checkpoint(model_dir, ["tbl19_20sc_scale_consistent.pt", "tbl19.pt"])
    if tbl_path is None:
        tbl_path = os.path.join(os.path.dirname(__file__), "src", "tbl19_20sc_scale_consistent.pt")

    if os.path.exists(tbl_path):
        tbl_data = torch.load(tbl_path, map_location="cpu", weights_only=False)
        base_table = tbl_data["table"] if isinstance(tbl_data, dict) and "table" in tbl_data else tbl_data
        print(f"  [Track 1] Loaded RSUIGM table from {tbl_path}", flush=True)
    else:
        base_table = torch.zeros(19, 20, 3)
        print("  [Track 1] WARNING: RSUIGM table not found, using dummy zeros.", flush=True)

    # 2. Load Refiner Models (5-Fold Ensemble)
    refiner_models = []
    for f in range(5):
        r_path = find_checkpoint(model_dir, [
            f"refiner_fold{f}.pt",
            f"refiner_f{f}.pt",
            f"fold_{f}/refiner_best.pt",
            f"fold{f}/refiner_best.pt"
        ])
        if r_path and os.path.exists(r_path):
            r_m = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(dev)
            sd = torch.load(r_path, map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            r_m.load_state_dict(sd)
            r_m.eval()
            refiner_models.append(r_m)
            print(f"  [Track 1] Loaded Refiner Fold {f}: {r_path}", flush=True)

    # Fallback to single refiner if 5 folds not found
    if not refiner_models:
        single_r = find_checkpoint(model_dir, ["refiner_best.pt", "refiner.pt"])
        if single_r and os.path.exists(single_r):
            r_m = build_v18_restoration_refiner(rsuigm_table=base_table, use_neural_refiner=True).to(dev)
            sd = torch.load(single_r, map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            r_m.load_state_dict(sd)
            r_m.eval()
            refiner_models.append(r_m)
            print(f"  [Track 1] Fallback: Loaded single refiner {single_r}", flush=True)
        else:
            raise FileNotFoundError(f"No refiner checkpoints found in {model_dir}")

    # 3. Load Segmenter Models (5-Fold Ensemble with Fold 3 Pilot Upgrade)
    segmenter_models = []
    for f in range(5):
        s_path = find_checkpoint(model_dir, [
            f"segmenter_fold{f}.pt",
            f"segmenter_f{f}.pt",
            f"fold_{f}/segmenter_best.pt",
            f"fold{f}/segmenter_best.pt"
        ])
        if s_path and os.path.exists(s_path):
            s_m = build_v18_segmenter(num_classes=NUM_CLASSES).to(dev)
            sd = torch.load(s_path, map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            s_m.load_state_dict(sd)
            s_m.eval()
            segmenter_models.append(s_m)
            print(f"  [Track 1] Loaded Segmenter Fold {f}: {s_path}", flush=True)

    # Fallback to single segmenter if 5 folds not found
    if not segmenter_models:
        single_s = find_checkpoint(model_dir, ["segmenter_best.pt", "segmenter.pt"])
        if single_s and os.path.exists(single_s):
            s_m = build_v18_segmenter(num_classes=NUM_CLASSES).to(dev)
            sd = torch.load(single_s, map_location=dev, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            s_m.load_state_dict(sd)
            s_m.eval()
            segmenter_models.append(s_m)
            print(f"  [Track 1] Fallback: Loaded single segmenter {single_s}", flush=True)
        else:
            raise FileNotFoundError(f"No segmenter checkpoints found in {model_dir}")

    print(f"[Track 1] Initialized {len(refiner_models)}-fold Refiners and {len(segmenter_models)}-fold Segmenters.", flush=True)

    preprocessor = V18CanonicalPreprocessor(max_side=MAX_SIDE, pad_multiple=32)
    cal_t = torch.tensor(OFFSETS, dtype=torch.float32, device=dev).view(1, NUM_CLASSES, 1, 1)

    names = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[Track 1] Processing {len(names)} images...", flush=True)

    use_amp = (dev.type == "cuda")

    for idx, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(input_dir, name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        orig_h, orig_w = img_bgr.shape[:2]
        wt, dp = parse_filename_meta(name)

        prep = preprocessor.prepare_input(img_bgr)
        raw_pad = prep["rgb_tensor"].to(dev)       # [1, 3, H_pad, W_pad]
        meta = prep["meta"]
        wt_t = torch.tensor([wt], dtype=torch.long, device=dev)
        dp_t = torch.tensor([dp], dtype=torch.float32, device=dev)

        with torch.no_grad():
            # A. 5-Fold Refiner Ensemble Forward
            restored_accum = torch.zeros_like(raw_pad)
            for r_m in refiner_models:
                out_r = r_m(raw_pad.float(), wt_t, dp_t)
                restored_accum += out_r["refined_image"]
            restored_fused = restored_accum / len(refiner_models)

            # Unpad restoration & detail enhancement
            restored_unpad = preprocessor.unpad_output(
                restored_fused.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                meta
            )
            blur = cv2.GaussianBlur(restored_unpad, (0, 0), SIGMA_RESTORATION)
            restored_final = np.clip(restored_unpad + ALPHA_RESTORATION * (restored_unpad - blur), 0.0, 1.0)
            rest_u8 = np.clip(restored_final * 255.0, 0, 255).astype(np.uint8)

            # Restored BGR representation with JPEG quality 98 parity
            rest_bgr = cv2.cvtColor(rest_u8, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
            _, encimg = cv2.imencode('.jpg', rest_bgr, encode_param)
            rest_bgr_jpeg = cv2.imdecode(encimg, 1)

            # Prepare restored input for segmentation
            rest_prep = preprocessor.prepare_input(rest_bgr_jpeg)
            rest_pad = rest_prep["rgb_tensor"].to(dev)

            # Horizontal flip TTA tensors
            raw_pad_flip = torch.flip(raw_pad, dims=[3])
            rest_pad_flip = torch.flip(rest_pad, dims=[3])

            # B. 5-Fold Segmenter Ensemble with Horizontal-Flip TTA
            prob_accum = torch.zeros((1, NUM_CLASSES, raw_pad.shape[2], raw_pad.shape[3]), device=dev)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for s_m in segmenter_models:
                        out_orig = s_m(rest_pad, raw_pad, wt_t, dp_t)
                        logits_orig = out_orig.get("logits", out_orig.get("cls_logits")) if isinstance(out_orig, dict) else out_orig
                        p_orig = F.softmax(logits_orig.float(), dim=1)

                        out_flip = s_m(rest_pad_flip, raw_pad_flip, wt_t, dp_t)
                        logits_flip = out_flip.get("logits", out_flip.get("cls_logits")) if isinstance(out_flip, dict) else out_flip
                        p_flip = torch.flip(F.softmax(logits_flip.float(), dim=1), dims=[3])

                        prob_accum += 0.5 * (p_orig + p_flip)
            else:
                for s_m in segmenter_models:
                    out_orig = s_m(rest_pad, raw_pad, wt_t, dp_t)
                    logits_orig = out_orig.get("logits", out_orig.get("cls_logits")) if isinstance(out_orig, dict) else out_orig
                    p_orig = F.softmax(logits_orig.float(), dim=1)

                    out_flip = s_m(rest_pad_flip, raw_pad_flip, wt_t, dp_t)
                    logits_flip = out_flip.get("logits", out_flip.get("cls_logits")) if isinstance(out_flip, dict) else out_flip
                    p_flip = torch.flip(F.softmax(logits_flip.float(), dim=1), dims=[3])

                    prob_accum += 0.5 * (p_orig + p_flip)

            prob_mean = prob_accum / len(segmenter_models)

            # C. Calibrated logit offset addition
            log_prob = torch.log(prob_mean + 1e-10) + cal_t
            mask_pad = torch.argmax(log_prob, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            mask_unpad = preprocessor.unpad_output(mask_pad, meta)

        # Target scoring size: max-side 1024, keeping aspect ratio
        tw, th = _scoring_size(orig_w, orig_h)
        if (orig_w, orig_h) != (tw, th):
            rest_final_resized = cv2.resize(rest_bgr_jpeg, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_unpad, (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            rest_final_resized = rest_bgr_jpeg
            mask_resized = mask_unpad

        # Save outputs matching official EURS format
        # Restored image (JPEG 98)
        out_res_path = os.path.join(output_dir, "restored", stem + ".jpg")
        cv2.imwrite(out_res_path, rest_final_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        # Segmentation mask (single-channel uint8 PNG)
        out_mask_path = os.path.join(output_dir, "seg_masks", stem + ".png")
        cv2.imwrite(out_mask_path, mask_resized)

        if (idx + 1) % 100 == 0 or idx == len(names) - 1:
            print(f"  Processed [{idx+1:04d}/{len(names)}] images...", flush=True)

    print("[Track 1] Inference completed successfully.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="weights")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.model_dir, args.input_dir, args.output_dir, args.device)

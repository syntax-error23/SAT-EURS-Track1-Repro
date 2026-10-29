"""
SAT-EURS Track 1 — V18 Canonical Preprocessing & Data Contracts
Implements strict aspect-preserving resize, internal multiple-of-32 padding,
cropping before serialization, condition parameter validation, and run manifest generation.
"""

import os
import os.path as osp
import json
import hashlib
from typing import Tuple, Dict, Any, Optional, List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VALID_WATER_TYPES = set(range(1, 11))
VALID_DEPTHS = set(range(5, 105, 5))
EURS_VALID_CLASSES = {0, 1, 2, 3, 4}
IGNORE_INDEX = 255


def parse_scene_wt_depth(fname: str) -> Tuple[str, int, int]:
    """
    Parses {scene}_{water_type}_{depth} from filename with strict range validation.
    Raises ValueError on any deviation.
    """
    stem = osp.splitext(osp.basename(fname))[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid EURS filename format (expected 'scene_wt_depth'): {fname}")
    
    scene = parts[0]
    try:
        wt = int(parts[1])
        depth = int(parts[2])
    except ValueError as e:
        raise ValueError(f"Non-integer condition parameters in '{fname}': {e}")

    if wt not in VALID_WATER_TYPES:
        raise ValueError(f"Water type {wt} out of valid EURS range [1..10] in '{fname}'")
    if depth not in VALID_DEPTHS:
        raise ValueError(f"Depth {depth} out of valid EURS discrete stations [5..100 step 5] in '{fname}'")

    return scene, wt, depth


def scale_to_max_side(w: int, h: int, max_side: int = 1024) -> Tuple[int, int]:
    """
    Official MaCVi aspect-preserving resize geometry helper.
    Ensures max(w, h) <= max_side without changing native aspect ratio.
    """
    scale = max_side / max(w, h)
    if scale >= 1.0:
        return w, h
    target_w = max(1, round(w * scale))
    target_h = max(1, round(h * scale))
    return target_w, target_h


class V18CanonicalPreprocessor:
    """
    Single canonical preprocessor used across training, validation, and inference.
    Guarantee: Aspect-preserving scaling + reflection padding to multiple of 32 +
    strict unpadding before output serialization.
    """

    def __init__(self, max_side: int = 1024, pad_multiple: int = 32):
        self.max_side = max_side
        self.pad_multiple = pad_multiple
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def prepare_input(
        self,
        image_bgr: np.ndarray,
        mask_u8: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Prepares raw image and optional mask.
        Returns:
            rgb_tensor: [1, 3, H_pad, W_pad] in [0, 1]
            norm_tensor: [1, 3, H_pad, W_pad] ImageNet normalized
            mask_tensor: [1, H_pad, W_pad] with discrete IDs or 255 for padded/ignored
            meta: geometry info for exact unpadding
        """
        orig_h, orig_w = image_bgr.shape[:2]
        target_w, target_h = scale_to_max_side(orig_w, orig_h, self.max_side)

        # 1. Aspect-preserving resize
        if (orig_w, orig_h) != (target_w, target_h):
            rgb_scaled = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if mask_u8 is not None:
                mask_scaled = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_scaled = None
        else:
            rgb_scaled = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mask_scaled = mask_u8.copy() if mask_u8 is not None else None

        if mask_scaled is not None:
            self.validate_mask(mask_scaled)

        # 2. Pad to multiple of pad_multiple (32)
        pad_h = (self.pad_multiple - target_h % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - target_w % self.pad_multiple) % self.pad_multiple

        # Reflection pad on RGB, constant (255) pad on mask
        if pad_h > 0 or pad_w > 0:
            rgb_padded = cv2.copyMakeBorder(
                rgb_scaled, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
            if mask_scaled is not None:
                mask_padded = cv2.copyMakeBorder(
                    mask_scaled, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX
                )
            else:
                mask_padded = None
        else:
            rgb_padded = rgb_scaled
            mask_padded = mask_scaled

        # 3. Convert to float32 [0, 1] tensor
        rgb_f32 = rgb_padded.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H_pad, W_pad]

        # Normalized tensor for semantic transformer encoders
        mean_t = torch.tensor(self.mean).view(1, 3, 1, 1)
        std_t = torch.tensor(self.std).view(1, 3, 1, 1)
        norm_tensor = (rgb_tensor - mean_t) / std_t

        mask_tensor = None
        if mask_padded is not None:
            mask_tensor = torch.from_numpy(mask_padded.astype(np.int64)).unsqueeze(0)  # [1, H_pad, W_pad]

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "target_h": target_h,
            "target_w": target_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "padded_h": target_h + pad_h,
            "padded_w": target_w + pad_w,
        }

        return {
            "rgb_tensor": rgb_tensor,
            "norm_tensor": norm_tensor,
            "mask_tensor": mask_tensor,
            "meta": meta
        }

    @staticmethod
    def unpad(
        output: Union[torch.Tensor, np.ndarray],
        target_h: int,
        target_w: int
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Strips internal multiple-of-32 padding, returning exact target resolution.
        """
        if isinstance(output, torch.Tensor):
            if output.ndim == 4:
                return output[:, :, :target_h, :target_w]
            elif output.ndim == 3:
                return output[:, :target_h, :target_w]
            elif output.ndim == 2:
                return output[:target_h, :target_w]
        elif isinstance(output, np.ndarray):
            if output.ndim == 3:
                return output[:target_h, :target_w, :]
            elif output.ndim == 2:
                return output[:target_h, :target_w]
        raise ValueError(f"Unsupported output type or dimension for unpad: {type(output)}")

    @staticmethod
    def unpad_output(
        output: Union[torch.Tensor, np.ndarray],
        meta: Dict[str, Any]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Strips internal multiple-of-32 padding using the geometry dictionary from prepare_input.
        """
        target_h, target_w = meta["target_h"], meta["target_w"]
        return V18CanonicalPreprocessor.unpad(output, target_h, target_w)

    @staticmethod
    def validate_mask(mask: np.ndarray, allow_ignore: bool = True):
        """
        Validates discrete categorical mask invariants.
        Only classes {0, 1, 2, 3, 4} (and 255 if allow_ignore=True) are permitted.
        """
        unique_classes = set(np.unique(mask).tolist())
        allowed = EURS_VALID_CLASSES | ({IGNORE_INDEX} if allow_ignore else set())
        invalid = unique_classes - allowed
        if invalid:
            raise ValueError(f"Mask contains invalid class IDs {invalid}. Permitted: {allowed}")

    @staticmethod
    def create_run_manifest(
        run_name: str,
        checkpoint_paths: Dict[str, str],
        restoration_table_id: str,
        calibration_offsets: List[float],
        tta_config: Dict[str, Any],
        out_manifest_path: str
    ) -> Dict[str, Any]:
        """
        Creates a complete cryptographic provenance manifest for a V18 execution.
        """
        ckpt_hashes = {}
        for k, p in checkpoint_paths.items():
            if osp.exists(p):
                with open(p, "rb") as f:
                    h = hashlib.sha256(f.read()).hexdigest()
                ckpt_hashes[k] = {"path": p, "sha256": h, "size_bytes": osp.getsize(p)}
            else:
                ckpt_hashes[k] = {"path": p, "status": "MISSING_OR_PLANNED"}

        manifest = {
            "version": "v18",
            "run_name": run_name,
            "restoration_table_id": restoration_table_id,
            "calibration_offsets": calibration_offsets,
            "tta_config": tta_config,
            "checkpoints": ckpt_hashes,
        }

        os.makedirs(osp.dirname(out_manifest_path), exist_ok=True)
        with open(out_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    @staticmethod
    def validate_fold_manifest_integrity(manifest_dict: Dict[str, Any]):
        """
        Asserts that all 200 condition variants from the same EURS scene stay strictly within one fold.
        Zero cross-split leakage allowed.
        """
        folds = manifest_dict.get("folds", {})
        assert len(folds) == 5, f"Expected exactly 5 folds, found {len(folds)}"

        all_scenes_seen = set()
        for f_id, f_data in folds.items():
            tr = set(f_data["train_scenes"])
            val = set(f_data["val_scenes"])
            overlap = tr.intersection(val)
            assert len(overlap) == 0, f"Disjoint split violation in {f_id}: overlap={overlap}"
            assert len(tr) == 16, f"Expected 16 train scenes in {f_id}, found {len(tr)}"
            assert len(val) == 4, f"Expected 4 val scenes in {f_id}, found {len(val)}"
            all_scenes_seen.update(tr)
            all_scenes_seen.update(val)

        assert len(all_scenes_seen) == 20, f"Expected 20 unique scenes total, found {len(all_scenes_seen)}"
        return True

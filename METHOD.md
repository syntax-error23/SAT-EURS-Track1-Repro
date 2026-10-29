# Method description

Team: SAT (MIT-BLR)   Track(s): Track 1 (eurs-t1)   Reported testing-phase submission(s): #19359 (Composite: 0.6410)

## Approach
`V33-Ensemble-F3Upgrade` is an unconstrained multi-task perception pipeline combining physical radiometric restoration with a high-capacity dual-stream semantic segmenter.
1. **Physical Restoration Engine (RSUIGM-19 + Bounded NAFNet):** Degraded pixels $[r,g,b]^T$ are expanded into a 19-dimensional polynomial basis and transformed via closed-form Tikhonov regularized matrices ($\lambda = 10^{-4}$). A 16-block Non-Linear Activation Free (NAF) network ingests the 8-channel tensor [Raw RGB, RSUIGM RGB, Water Type, Depth] bounded to $\pm 0.10 \times \tanh$, followed by Laplacian unsharp edge filtering ($\alpha=1.34, \sigma=1.5$), reaching 25.48 dB PSNR and 0.829 SSIM.
2. **Dual-Stream Segmenter (`EURSAPEXV18Segmenter`):** Resolves the 15-pixel ViT boundary dilation via two parallel asymmetric backbones:
   - *Semantic Stream:* DINOv3 ViT-Large/16 (304M params) ingesting restored RGB at $518\times518$ for invariant global semantics.
   - *Inductive Stream:* ConvNeXt-Base (88.5M params) ingesting raw degraded RGB at $512\times512$ with a 2-stage DetailStem to preserve fine micro-appendages.
   - *Fusion Neck:* Learned spatial gating $f_{\text{fused}} = \sigma(G) \odot f_{\text{sem}} + (1 - \sigma(G)) \odot f_{\text{conv}}$ with condition FiLM modulation in a 4-level UPerNet neck.
3. **5-Fold 'F3Upgrade' Ensemble:** Combines Folds 0, 1, 2, 4 with a retrained Fold 3 pilot checkpoint specifically optimized for rare elasmobranch recall via dynamic copy-paste oversampling (+0.1636 $Q_{\text{seg}}$ OOF gain).
4. **Logit Calibration:** Adds calibrated static offset $\Delta Z = [0.0, 0.0, 0.0, +0.10, +0.25]$ prior to $\operatorname{argmax}$, defeating the hard Zero-True-Positive coverage penalty cliff on rare classes.

## Training data
Trained exclusively on the 20 provided EURS training scenes across 5 stratified folds. The 6 development scenes were reserved strictly for validation, hyperparameter selection, and discrepancy audits. Zero testing-phase images were used in training, fine-tuning, adaptation, or pseudo-labelling.
- **Exact Training Scenes (20 scenes):** `fish1`, `fish2`, `fish4`, `fish5`, `fish7`, `fish9`, `fish11`, `fish12`, `fish13`, `fish14`, `fish15`, `fish17`, `fish18`, `fish20`, `fish21`, `ray1`, `seahorse1`, `sharkFish`, `starfish2`, `turtles1`.
- **Development Scenes (6 scenes, validation only):** `fish3`, `fish6`, `fish10`, `fish19`, `sharRayFish`, `starfish1`.
- **External Data:** None.
- **Pretrained Models:** DINOv3 ViT-Large/16 (LVD-142M self-supervised), ConvNeXt-Base (ImageNet-1k).

## Anything non-standard at inference
- Dual outputs: `restored/<stem>.png` (RGB, max side 1024px) and `seg_masks/<stem>.png` (uint8 mask in $\{0, 1, 2, 3, 4\}$, max side 1024px).
- Test-Time Augmentation (TTA): Horizontal flip averaged across all 5 folds (10 forward passes total).
- Calibrated logit offsets $[0.0, 0.0, 0.0, +0.10, +0.25]$ added before $\operatorname{argmax}$.

## Reproduction notes
- **Hardware:** Single NVIDIA GeForce RTX 5090 (32GB VRAM) or RTX 3090 / 4090 (24GB).
- **Runtime:** $\approx 20$ minutes for 1,200 testing images.
- **Weights Link:** Public archive detailed in `weights/weights_link.yaml`.

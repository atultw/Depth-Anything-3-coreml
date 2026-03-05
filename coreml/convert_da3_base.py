#!/usr/bin/env python3
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
CoreML conversion for DA3-Base: Pose-Conditioned Depth Estimation.

Converts the DepthAnything3 da3-base model to a CoreML .mlpackage that
accepts **N** input images with camera extrinsics and intrinsics and
outputs depth maps, confidence maps, and predicted camera parameters.

Usage
-----
    python coreml/convert_da3_base.py \\
        --output DA3Base.mlpackage \\
        --height 504 --width 504 \\
        --max-views 32

Prerequisites
-------------
    pip install coremltools>=7.0 torch>=2.0

Assumptions
-----------
* Batch dimension B is always 1 (one scene per call).
* Spatial dimensions (H, W) are *fixed* at conversion time and must each
  be divisible by 14 (the ViT patch size).
* The number of views **N** is flexible in [min_views, max_views].
* Input images must already be resized to (H, W) and normalised with
  ImageNet statistics  mean=[0.485, 0.456, 0.406]  std=[0.229, 0.224, 0.225].
* Extrinsics are (N, 4, 4) world-to-camera matrices — the model normalises
  them internally (first-view → identity, median-distance → 1).
* Intrinsics are (N, 3, 3) matrices already adjusted for the processing
  resolution (fx, fy, cx, cy scaled to H×W pixel space).
* Only the pose-conditioned depth path is exported — no Gaussian-Splatting,
  feature export, GLB export, or ray-based pose estimation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make sure the repository source is importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.geometry import affine_inverse  # noqa: E402


# ======================================================================== #
#  Traceable wrapper                                                        #
# ======================================================================== #
class DA3BaseCoreMLWrapper(nn.Module):
    """Thin, trace-friendly wrapper around :class:`DepthAnything3Net`.

    It encapsulates:

    1. **Extrinsic normalisation** — first view becomes identity and the
       median camera distance is normalised to 1, reproducing the logic in
       :pymethod:`DepthAnything3._normalize_extrinsics`.
    2. **Model forward pass** — backbone, depth head, camera decoder.  GS,
       ray-pose and feature-export branches are disabled.
    3. **Output unpacking** — converts the internal ``addict.Dict`` to a
       plain tuple of tensors so ``torch.jit.trace`` can record it.

    Inputs (all ``float32``):

    =========== ============== ==========================================
    Name        Shape          Description
    =========== ============== ==========================================
    images      (1, N, 3, H, W) ImageNet-normalised images
    extrinsics  (1, N, 4, 4)    world-to-camera (raw, normalised inside)
    intrinsics  (1, N, 3, 3)    adjusted for processing resolution
    =========== ============== ==========================================

    Outputs (all ``float32``):

    ================ ============== ====================================
    Name             Shape          Description
    ================ ============== ====================================
    depth            (1, N, H, W)   estimated depth (positive, ``exp``)
    confidence       (1, N, H, W)   depth confidence (``exp + 1``)
    pred_extrinsics  (1, N, 3, 4)   predicted w2c (compact)
    pred_intrinsics  (1, N, 3, 3)   predicted camera intrinsics
    ================ ============== ====================================
    """

    def __init__(self, da3_net: nn.Module) -> None:
        super().__init__()
        self.net = da3_net

    # ------------------------------------------------------------------ #
    #  forward                                                             #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ext_norm = self._normalize_extrinsics(extrinsics)

        output = self.net(
            images,
            ext_norm,
            intrinsics,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )

        return (
            output["depth"],       # (1, N, H, W)
            output["depth_conf"],  # (1, N, H, W)
            output["extrinsics"],  # (1, N, 3, 4)
            output["intrinsics"],  # (1, N, 3, 3)
        )

    # ------------------------------------------------------------------ #
    #  extrinsic normalisation (no in-place ops, trace-safe)               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_extrinsics(ext: torch.Tensor) -> torch.Tensor:
        """Normalise so that view-0 is identity and median distance ≈ 1."""
        transform = affine_inverse(ext[:, :1])          # (1, 1, 4, 4)
        ext_norm = ext @ transform                      # relative to first view

        c2ws = affine_inverse(ext_norm)
        translations = c2ws[..., :3, 3]                 # (1, N, 3)
        dists = translations.norm(dim=-1)               # (1, N)
        median_dist = torch.clamp(torch.median(dists), min=0.1)

        # Build output without in-place mutation (trace-safe)
        rot_part = ext_norm[..., :3, :3]                # (1, N, 3, 3)
        trans_part = ext_norm[..., :3, 3:] / median_dist  # (1, N, 3, 1)
        bottom = ext_norm[..., 3:, :]                   # (1, N, 1, 4)
        top = torch.cat([rot_part, trans_part], dim=-1) # (1, N, 3, 4)
        return torch.cat([top, bottom], dim=-2)         # (1, N, 4, 4)


# ======================================================================== #
#  CLI helpers                                                              #
# ======================================================================== #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert DA3-Base (pose-conditioned) to CoreML"
    )
    p.add_argument(
        "--output", type=str, default="DA3Base.mlpackage",
        help="Output path for the .mlpackage (default: DA3Base.mlpackage)",
    )
    p.add_argument(
        "--height", type=int, default=504,
        help="Fixed image height; must be divisible by 14 (default: 504)",
    )
    p.add_argument(
        "--width", type=int, default=504,
        help="Fixed image width; must be divisible by 14 (default: 504)",
    )
    p.add_argument(
        "--min-views", type=int, default=1,
        help="Minimum number of input views (default: 1)",
    )
    p.add_argument(
        "--max-views", type=int, default=32,
        help="Maximum number of input views (default: 32)",
    )
    p.add_argument(
        "--trace-views", type=int, default=2,
        help="Number of views used during tracing (default: 2)",
    )
    p.add_argument(
        "--model-name", type=str, default="da3-base",
        help="Model name key in the registry (default: da3-base)",
    )
    p.add_argument(
        "--model-id", type=str, default=None,
        help=(
            "HuggingFace model-id to load via from_pretrained(). "
            "When given, --model-name is only used for metadata."
        ),
    )
    return p.parse_args()


# ======================================================================== #
#  Main conversion pipeline                                                 #
# ======================================================================== #
def main() -> None:
    import coremltools as ct  # imported here so the rest is parseable w/o ct

    args = _parse_args()
    H, W = args.height, args.width

    # --- sanity checks ---------------------------------------------------
    if H % 14 != 0 or W % 14 != 0:
        sys.exit(f"ERROR: height ({H}) and width ({W}) must be divisible by 14.")
    if args.trace_views < args.min_views or args.trace_views > args.max_views:
        sys.exit(
            f"ERROR: --trace-views ({args.trace_views}) must be in "
            f"[{args.min_views}, {args.max_views}]."
        )

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"[1/5] Loading model ...")
    if args.model_id is not None:
        model = DepthAnything3.from_pretrained(args.model_id)
    else:
        model = DepthAnything3(model_name=args.model_name)
    model.eval()

    da3_net = model.model       # inner DepthAnything3Net
    da3_net.eval()

    # ------------------------------------------------------------------
    # 2. Build wrapper
    # ------------------------------------------------------------------
    wrapper = DA3BaseCoreMLWrapper(da3_net).eval()

    # Move to CPU (CoreML tracing must run on CPU)
    wrapper = wrapper.float().cpu()

    # ------------------------------------------------------------------
    # 3. Trace
    # ------------------------------------------------------------------
    N_trace = args.trace_views
    print(f"[2/5] Tracing with N={N_trace}, H={H}, W={W} ...")

    example_images = torch.randn(1, N_trace, 3, H, W)

    # Non-degenerate example extrinsics (slight translations per view)
    example_ext = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, N_trace, 1, 1)
    for i in range(1, N_trace):
        example_ext[0, i, 0, 3] = float(i) * 0.1

    # Reasonable intrinsics (focal ~ image size, principal point at centre)
    example_int = torch.zeros(1, N_trace, 3, 3)
    example_int[..., 0, 0] = float(W)       # fx
    example_int[..., 1, 1] = float(H)       # fy
    example_int[..., 0, 2] = float(W) / 2   # cx
    example_int[..., 1, 2] = float(H) / 2   # cy
    example_int[..., 2, 2] = 1.0

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (example_images, example_ext, example_int),
            strict=False,
        )

    # Quick forward-pass sanity check
    print("[3/5] Validating traced model ...")
    with torch.no_grad():
        depth, conf, p_ext, p_int = traced(example_images, example_ext, example_int)
    print(f"       depth        : {tuple(depth.shape)}")
    print(f"       confidence   : {tuple(conf.shape)}")
    print(f"       pred_ext     : {tuple(p_ext.shape)}")
    print(f"       pred_int     : {tuple(p_int.shape)}")

    # ------------------------------------------------------------------
    # 4. Convert to CoreML
    # ------------------------------------------------------------------
    print("[4/5] Converting to CoreML (mlprogram) ...")

    n_dim = ct.RangeDim(
        lower_bound=args.min_views,
        upper_bound=args.max_views,
        default=args.trace_views,
    )

    ct_inputs = [
        ct.TensorType(
            name="images",
            shape=(1, n_dim, 3, H, W),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="extrinsics",
            shape=(1, n_dim, 4, 4),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="intrinsics",
            shape=(1, n_dim, 3, 3),
            dtype=np.float32,
        ),
    ]

    ct_outputs = [
        ct.TensorType(name="depth"),
        ct.TensorType(name="confidence"),
        ct.TensorType(name="pred_extrinsics"),
        ct.TensorType(name="pred_intrinsics"),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )

    # ---- metadata -------------------------------------------------------
    mlmodel.author = "DepthAnything3"
    mlmodel.short_description = (
        f"DA3-Base pose-conditioned depth estimation.  "
        f"Accepts {args.min_views}–{args.max_views} images at {H}×{W}."
    )
    mlmodel.input_description["images"] = (
        f"ImageNet-normalised images, shape (1, N, 3, {H}, {W})."
    )
    mlmodel.input_description["extrinsics"] = (
        "World-to-camera extrinsic matrices, shape (1, N, 4, 4)."
    )
    mlmodel.input_description["intrinsics"] = (
        "Camera intrinsic matrices (adjusted for processing resolution), "
        "shape (1, N, 3, 3)."
    )
    mlmodel.output_description["depth"] = (
        f"Estimated depth maps, shape (1, N, {H}, {W})."
    )
    mlmodel.output_description["confidence"] = (
        f"Depth confidence maps, shape (1, N, {H}, {W})."
    )
    mlmodel.output_description["pred_extrinsics"] = (
        "Predicted world-to-camera extrinsics (compact), shape (1, N, 3, 4)."
    )
    mlmodel.output_description["pred_intrinsics"] = (
        "Predicted camera intrinsics, shape (1, N, 3, 3)."
    )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print(f"[5/5] Saving to {args.output} ...")
    mlmodel.save(args.output)
    print("✅  Conversion complete.")


if __name__ == "__main__":
    main()

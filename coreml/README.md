# CoreML Conversion — DA3-Base Pose-Conditioned Depth Estimation

Convert the **DA3-Base** model to a CoreML `.mlpackage` that accepts **N**
images (with camera extrinsics and intrinsics) in a single forward pass and
returns depth maps, confidence maps, and predicted camera parameters.
A companion **Swift wrapper** takes `CGImage` arrays and returns a combined
3-D point cloud in world space.

---

## Quick start

### 1. Convert (Python)

```bash
# install conversion dependency (in addition to the project requirements)
pip install coremltools>=7.0

# run conversion (downloads weights from HuggingFace the first time)
python coreml/convert_da3_base.py \
    --model-id depth-anything/DA3-Base \
    --output DA3Base.mlpackage \
    --height 504 --width 504 \
    --min-views 1 --max-views 32
```

The script:
1. Loads the DA3-Base model (via `from_pretrained` or from the local registry).
2. Wraps it in a trace-friendly module that handles extrinsic normalisation.
3. Traces with `torch.jit.trace`.
4. Converts to an ML Program (`mlprogram`) targeting **iOS 17+**.
5. Saves the `.mlpackage`.

### 2. Use in Swift

```swift
import CoreML

// Load (compile once, then cache the .mlmodelc)
let da3 = try DepthAnything3(packageURL: Bundle.main.url(forResource: "DA3Base", withExtension: "mlpackage")!)

// Prepare inputs
let images: [CGImage] = ...          // N images, any resolution
let extrinsics: [[Float]] = ...      // N × 16, row-major 4×4 w2c
let intrinsics: [[Float]] = ...      // N × 9,  row-major 3×3 (original image res)

// Run inference
let result = try da3.inference(
    images: images,
    extrinsics: extrinsics,
    intrinsics: intrinsics,
    confidenceThreshold: 40   // drop bottom 40 % by confidence
)

// Use outputs
print(result.pointCloud.count, "world-space points")
print(result.depthMaps.count, "depth maps")
```

---

## File overview

| File | Description |
|------|-------------|
| `convert_da3_base.py` | Python script – traces the PyTorch model and converts to CoreML. |
| `DepthAnything3.swift` | Swift wrapper – image preprocessing, model execution, Umeyama alignment, depth-to-point-cloud unprojection. |
| `README.md` | This file. |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  CoreML Model                    │
│                                                  │
│  inputs                                          │
│    images      (1, N, 3, H, W)   float32         │
│    extrinsics  (1, N, 4, 4)      float32         │
│    intrinsics  (1, N, 3, 3)      float32         │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │ Extrinsic normalisation                  │    │
│  │  • first view → identity                 │    │
│  │  • median camera distance → 1            │    │
│  └──────────────────────────────────────────┘    │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐    │
│  │ CameraEnc  (ext + int → camera tokens)   │    │
│  └────────┬─────────────────────────────────┘    │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐    │
│  │ DinoV2 ViT-B backbone                    │    │
│  │  • local + global alternating attention   │    │
│  │  • camera tokens injected at alt_start    │    │
│  └────────┬─────────────────────────────────┘    │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐    │
│  │ DualDPT depth head                       │    │
│  │  → depth (1,N,H,W), conf (1,N,H,W)      │    │
│  └────────┬─────────────────────────────────┘    │
│           │                                      │
│  ┌────────▼─────────────────────────────────┐    │
│  │ CameraDec                                │    │
│  │  → pred_extrinsics (1,N,3,4)            │    │
│  │  → pred_intrinsics (1,N,3,3)            │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  outputs                                         │
│    depth, confidence, pred_extrinsics,           │
│    pred_intrinsics                               │
└──────────────────────────────────────────────────┘
            │
            ▼  (Swift wrapper post-processing)
┌──────────────────────────────────────────────────┐
│ 1. Umeyama Sim(3) alignment  →  scale factor     │
│ 2. Depth rescaling:  depth /= scale              │
│ 3. Per-pixel unprojection to world space         │
│    p_cam = K⁻¹ · [u, v, 1]ᵀ · d                 │
│    p_world = c2w · [p_cam, 1]ᵀ                   │
│ 4. Combined N-view point cloud returned          │
└──────────────────────────────────────────────────┘
```

---

## Assumptions & limitations

| # | Item |
|---|------|
| 1 | **Batch size** is always 1 (one scene per inference call). |
| 2 | **Spatial resolution** (H, W) is fixed at conversion time and must be divisible by 14. Default: 504 × 504. |
| 3 | **Number of views N** is flexible within `[min_views, max_views]` (via `ct.RangeDim`). |
| 4 | Images must be **preprocessed** before the CoreML model sees them: resized to (H, W) and normalised with ImageNet mean/std. The Swift wrapper performs this automatically. |
| 5 | **Intrinsics** provided to the Swift wrapper should correspond to the **original** image resolution — the wrapper adjusts `fx, fy, cx, cy` for the resize to (H, W). |
| 6 | The **extrinsic normalisation** (first-view identity, median-distance scaling) is embedded in the CoreML model. |
| 7 | After inference the Swift wrapper performs **Umeyama alignment** between predicted and input camera centres to compute a scale factor; depth is then rescaled by `1/scale`, and the **input** extrinsics are used for unprojection. This mirrors the `align_to_input_ext_scale=True` behaviour of the Python API. |
| 8 | Only the **pose-conditioned depth** path is exported. No Gaussian-Splatting head, feature export, GLB/NPZ export, or ray-based pose estimation. |
| 9 | The model targets **iOS 17+ / macOS 14+** (`ct.target.iOS17`). |
| 10 | `torch.jit.trace` is used for conversion; all data-dependent branches (reference-view selection, GS head) are inactive in the pose-conditioned path so the trace is faithful. |

---

## Conversion CLI reference

```
python coreml/convert_da3_base.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `DA3Base.mlpackage` | Output path |
| `--height` | 504 | Fixed image height (÷14) |
| `--width` | 504 | Fixed image width (÷14) |
| `--min-views` | 1 | Min N for `RangeDim` |
| `--max-views` | 32 | Max N for `RangeDim` |
| `--trace-views` | 2 | N used during `torch.jit.trace` |
| `--model-name` | `da3-base` | Registry key (local configs) |
| `--model-id` | *(none)* | HuggingFace model id (`from_pretrained`) |

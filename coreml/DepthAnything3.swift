// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
// SPDX-License-Identifier: Apache-2.0
//
// DepthAnything3.swift
// CoreML wrapper for DA3-Base pose-conditioned depth estimation.
//
// Accepts N CGImages together with their camera extrinsics (world-to-camera,
// 4×4) and intrinsics (3×3) and returns a combined 3-D point cloud in world
// space plus the aligned camera parameters.
//
// Assumptions
// -----------
// * The CoreML model was exported with convert_da3_base.py at a fixed
//   spatial resolution (H × W, default 504 × 504, both divisible by 14).
// * Images are resized & centre-cropped to that resolution internally.
// * Intrinsics are automatically adjusted for the resize / crop.
// * Extrinsic normalisation is done inside the CoreML model.
// * The Umeyama Sim(3) alignment is re-implemented in Swift (Accelerate).
// * Point-cloud unprojection uses the INPUT extrinsics/intrinsics after
//   depth has been rescaled by the Umeyama inverse scale.

import Accelerate
import CoreGraphics
import CoreML
import CoreImage

// MARK: - Public data types ------------------------------------------------

/// A single coloured 3-D point.
public struct ColoredPoint {
    public var position: (x: Float, y: Float, z: Float)
    public var color: (r: Float, g: Float, b: Float) // 0-1 range
}

/// Result returned by ``DepthAnything3/inference(images:extrinsics:intrinsics:confidenceThreshold:)``.
public struct DA3Result {
    /// Combined point cloud from all views (world space).
    public let pointCloud: [ColoredPoint]
    /// Per-view depth maps (row-major, H × W each).
    public let depthMaps: [[Float]]
    /// Per-view confidence maps (row-major, H × W each).
    public let confidenceMaps: [[Float]]
    /// Aligned extrinsics (world-to-camera), 16 floats (row-major 4×4) per view.
    public let extrinsics: [[Float]]
    /// Camera intrinsics, 9 floats (row-major 3×3) per view.
    public let intrinsics: [[Float]]
    /// Processing width used by the model.
    public let width: Int
    /// Processing height used by the model.
    public let height: Int
}

/// Errors specific to ``DepthAnything3``.
public enum DA3Error: Error, CustomStringConvertible {
    case modelLoadFailed(String)
    case outputExtractionFailed
    case imagePrepFailed(Int)

    public var description: String {
        switch self {
        case .modelLoadFailed(let msg):  return "Model load failed: \(msg)"
        case .outputExtractionFailed:    return "Could not extract model outputs"
        case .imagePrepFailed(let idx):  return "Failed to prepare image at index \(idx)"
        }
    }
}

// MARK: - DepthAnything3 wrapper -------------------------------------------

/// CoreML wrapper for DA3-Base pose-conditioned depth estimation.
///
/// ```swift
/// let da3 = try DepthAnything3(compiledModelURL: url)
/// let result = try da3.inference(
///     images: cgImages,
///     extrinsics: extrinsics4x4,   // [[Float]], each 16 elements (row-major 4×4)
///     intrinsics: intrinsics3x3    // [[Float]], each  9 elements (row-major 3×3)
/// )
/// print(result.pointCloud.count, "points")
/// ```
public final class DepthAnything3 {

    private let model: MLModel
    /// Fixed spatial dimensions the model was compiled with.
    public let modelWidth: Int
    public let modelHeight: Int

    // ImageNet normalisation constants.
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std:  [Float] = [0.229, 0.224, 0.225]

    // MARK: Init

    /// Load from a **compiled** Core ML model (`.mlmodelc`) directory.
    ///
    /// - Parameters:
    ///   - compiledModelURL: URL to the `.mlmodelc` on disk.
    ///   - width:  Processing width  (must match conversion `--width`).
    ///   - height: Processing height (must match conversion `--height`).
    public init(
        compiledModelURL: URL,
        width: Int = 504,
        height: Int = 504
    ) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let m = try? MLModel(contentsOf: compiledModelURL, configuration: config) else {
            throw DA3Error.modelLoadFailed(compiledModelURL.path)
        }
        self.model = m
        self.modelWidth = width
        self.modelHeight = height
    }

    /// Convenience: compile an `.mlpackage` at runtime, then load.
    ///
    /// - Parameters:
    ///   - packageURL: URL to the `.mlpackage` bundle.
    ///   - width:  Processing width.
    ///   - height: Processing height.
    public convenience init(
        packageURL: URL,
        width: Int = 504,
        height: Int = 504
    ) throws {
        let compiled = try MLModel.compileModel(at: packageURL)
        try self.init(compiledModelURL: compiled, width: width, height: height)
    }

    // MARK: Inference

    /// Run pose-conditioned depth estimation on **N** images.
    ///
    /// - Parameters:
    ///   - images:     Array of **N** ``CGImage`` instances (any size, resized internally).
    ///   - extrinsics: `N` world-to-camera 4×4 matrices – each element is a
    ///                 row-major `[Float]` of length 16.
    ///   - intrinsics: `N` camera intrinsic 3×3 matrices – each element is a
    ///                 row-major `[Float]` of length 9.  Provide the intrinsics
    ///                 matching the **original** image resolution; the wrapper
    ///                 adjusts them for the processing resolution internally.
    ///   - confidenceThreshold: Points below this percentile of the confidence
    ///                          distribution are discarded (default 0, keep all).
    /// - Returns: A ``DA3Result`` containing the combined point cloud and
    ///            per-view outputs.
    public func inference(
        images: [CGImage],
        extrinsics: [[Float]],
        intrinsics: [[Float]],
        confidenceThreshold: Float = 0
    ) throws -> DA3Result {
        let N = images.count
        precondition(extrinsics.count == N, "extrinsics count must equal image count")
        precondition(intrinsics.count == N, "intrinsics count must equal image count")
        precondition(extrinsics.allSatisfy { $0.count == 16 }, "each extrinsic: 16 floats")
        precondition(intrinsics.allSatisfy { $0.count == 9 },  "each intrinsic: 9 floats")

        let H = modelHeight
        let W = modelWidth

        // 1. Preprocess images & adjust intrinsics for the resize/crop.
        var adjustedIntrinsics: [[Float]] = []
        let imagesTensor = try prepareImagesTensor(
            images: images, W: W, H: H,
            originalIntrinsics: intrinsics,
            adjustedIntrinsicsOut: &adjustedIntrinsics
        )

        // 2. Prepare extrinsics tensor  (1, N, 4, 4)
        let extTensor = try matrix4x4Tensor(extrinsics, N: N)

        // 3. Prepare intrinsics tensor  (1, N, 3, 3)
        let intTensor = try matrix3x3Tensor(adjustedIntrinsics, N: N)

        // 4. Run the model.
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "images":     MLFeatureValue(multiArray: imagesTensor),
            "extrinsics": MLFeatureValue(multiArray: extTensor),
            "intrinsics": MLFeatureValue(multiArray: intTensor),
        ])
        let output = try model.prediction(from: input)

        guard
            let depthMA = output.featureValue(for: "depth")?.multiArrayValue,
            let confMA  = output.featureValue(for: "confidence")?.multiArrayValue,
            let pExtMA  = output.featureValue(for: "pred_extrinsics")?.multiArrayValue,
            let _       = output.featureValue(for: "pred_intrinsics")?.multiArrayValue
        else {
            throw DA3Error.outputExtractionFailed
        }

        // 5. Extract flat Float arrays.
        let depthFlat = Self.multiArrayToFloats(depthMA)  // 1*N*H*W
        let confFlat  = Self.multiArrayToFloats(confMA)
        let predExtFlat = Self.multiArrayToFloats(pExtMA) // 1*N*3*4 = N*12

        // 6. Umeyama alignment → scale.
        let scale = Self.umeyamaScale(
            inputExtrinsics4x4: extrinsics,
            predictedExtrinsics3x4: predExtFlat,
            N: N
        )

        // 7. Per-view: scale depth, unproject, collect point cloud.
        var allPoints: [ColoredPoint] = []
        var depthMaps:  [[Float]] = []
        var confMaps:   [[Float]] = []
        let pixelsPerView = H * W

        for v in 0..<N {
            let off = v * pixelsPerView
            var viewDepth = Array(depthFlat[off ..< off + pixelsPerView])
            let viewConf  = Array(confFlat[off ..< off + pixelsPerView])

            // Rescale depth by inverse Umeyama scale.
            if abs(scale) > 1e-8 {
                let invScale = 1.0 / scale
                vDSP.multiply(invScale, viewDepth, result: &viewDepth)
            }

            depthMaps.append(viewDepth)
            confMaps.append(viewConf)

            // Confidence-based filtering threshold (percentile of this view).
            let confThresh = Self.percentile(viewConf, p: confidenceThreshold)

            // Unproject using the original INPUT extrinsics & adjusted intrinsics.
            let pts = Self.unprojectToWorld(
                depth: viewDepth,
                confidence: viewConf,
                confThreshold: confThresh,
                extrinsic4x4: extrinsics[v],
                intrinsic3x3: adjustedIntrinsics[v],
                image: images[v],
                W: W, H: H
            )
            allPoints.append(contentsOf: pts)
        }

        return DA3Result(
            pointCloud: allPoints,
            depthMaps: depthMaps,
            confidenceMaps: confMaps,
            extrinsics: extrinsics,
            intrinsics: adjustedIntrinsics,
            width: W,
            height: H
        )
    }

    // MARK: - Image preprocessing -------------------------------------------

    /// Resize each CGImage to (W, H), normalise with ImageNet stats, and
    /// pack into an ``MLMultiArray`` of shape `(1, N, 3, H, W)`.
    /// Also outputs adjusted intrinsics for each view.
    private func prepareImagesTensor(
        images: [CGImage],
        W: Int, H: Int,
        originalIntrinsics: [[Float]],
        adjustedIntrinsicsOut: inout [[Float]]
    ) throws -> MLMultiArray {
        let N = images.count
        let arr = try MLMultiArray(shape: [1, N, 3, H, W] as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: N * 3 * H * W)

        for i in 0..<N {
            let img = images[i]
            let origW = img.width
            let origH = img.height

            // Resize to exactly (W, H).
            guard let ctx = CGContext(
                data: nil, width: W, height: H,
                bitsPerComponent: 8, bytesPerRow: W * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { throw DA3Error.imagePrepFailed(i) }
            ctx.interpolationQuality = .high
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
            guard let resized = ctx.makeImage(),
                  let data = resized.dataProvider?.data,
                  let bytes = CFDataGetBytePtr(data)
            else { throw DA3Error.imagePrepFailed(i) }

            // Fill tensor: NCHW layout, normalised with ImageNet stats.
            let base = i * 3 * H * W
            for c in 0..<3 {
                let mean = Self.mean[c]
                let std  = Self.std[c]
                for y in 0..<H {
                    for x in 0..<W {
                        let pixIdx = y * W + x
                        let byteIdx = pixIdx * 4 + c       // RGBX layout
                        let val = (Float(bytes[byteIdx]) / 255.0 - mean) / std
                        ptr[base + c * H * W + y * W + x] = val
                    }
                }
            }

            // Adjust intrinsics for the resize.
            // Row-major 3×3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            var K = originalIntrinsics[i]
            let sx = Float(W) / Float(origW)
            let sy = Float(H) / Float(origH)
            K[0] *= sx             // fx
            K[2] *= sx             // cx
            K[4] *= sy             // fy
            K[5] *= sy             // cy
            adjustedIntrinsicsOut.append(K)
        }

        return arr
    }

    // MARK: - Tensor helpers ------------------------------------------------

    private func matrix4x4Tensor(_ mats: [[Float]], N: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, N, 4, 4] as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: N * 16)
        for i in 0..<N {
            for j in 0..<16 { ptr[i * 16 + j] = mats[i][j] }
        }
        return arr
    }

    private func matrix3x3Tensor(_ mats: [[Float]], N: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, N, 3, 3] as [NSNumber], dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: N * 9)
        for i in 0..<N {
            for j in 0..<9 { ptr[i * 9 + j] = mats[i][j] }
        }
        return arr
    }

    static func multiArrayToFloats(_ ma: MLMultiArray) -> [Float] {
        let count = ma.count
        let ptr = ma.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    // MARK: - Umeyama alignment (scale only) --------------------------------

    /// Compute the Umeyama Sim(3) scale that aligns predicted camera centres
    /// to the input ones.  Only the **scale** is needed because the Swift side
    /// uses the original input extrinsics for unprojection.
    ///
    /// Algorithm:
    ///   1. Extract camera centres from w2c: C = -R^T t.
    ///   2. Compute centroids μ_in, μ_pred.
    ///   3. Compute cross-covariance H = Σ (pred'_i · in'_i^T) / N.
    ///   4. SVD(H) = U S V^T.
    ///   5. s = tr(S D) / σ²_pred  where D = diag(1,1,det(UV^T)).
    static func umeyamaScale(
        inputExtrinsics4x4: [[Float]],
        predictedExtrinsics3x4: [Float],  // flat, N*12
        N: Int
    ) -> Float {
        guard N >= 2 else { return 1.0 }

        // --- camera centres -------------------------------------------------
        func centre4x4(_ ext: [Float]) -> (Float, Float, Float) {
            // R = ext[0..9], t = ext[3,7,11] (row-major 4×4)
            let r00 = ext[0], r01 = ext[1], r02 = ext[2]
            let r10 = ext[4], r11 = ext[5], r12 = ext[6]
            let r20 = ext[8], r21 = ext[9], r22 = ext[10]
            let tx = ext[3], ty = ext[7], tz = ext[11]
            // C = -R^T * t
            let cx = -(r00*tx + r10*ty + r20*tz)
            let cy = -(r01*tx + r11*ty + r21*tz)
            let cz = -(r02*tx + r12*ty + r22*tz)
            return (cx, cy, cz)
        }

        func centre3x4(_ base: Int, _ buf: [Float]) -> (Float, Float, Float) {
            // Row-major 3×4 layout (stride 4 per row):
            //   Row 0 [base+0 .. base+3]:  r00  r01  r02  tx
            //   Row 1 [base+4 .. base+7]:  r10  r11  r12  ty
            //   Row 2 [base+8 .. base+11]: r20  r21  r22  tz
            let r00 = buf[base],   r01 = buf[base+1], r02 = buf[base+2]
            let r10 = buf[base+4], r11 = buf[base+5], r12 = buf[base+6]
            let r20 = buf[base+8], r21 = buf[base+9], r22 = buf[base+10]
            let tx = buf[base+3], ty = buf[base+7], tz = buf[base+11]
            let cx = -(r00*tx + r10*ty + r20*tz)
            let cy = -(r01*tx + r11*ty + r21*tz)
            let cz = -(r02*tx + r12*ty + r22*tz)
            return (cx, cy, cz)
        }

        var inPts  = [(Float, Float, Float)]()
        var prPts  = [(Float, Float, Float)]()
        for i in 0..<N {
            inPts.append(centre4x4(inputExtrinsics4x4[i]))
            prPts.append(centre3x4(i * 12, predictedExtrinsics3x4))
        }

        // --- centroids -------------------------------------------------------
        var muIn  = (Float(0), Float(0), Float(0))
        var muPr  = (Float(0), Float(0), Float(0))
        for i in 0..<N {
            muIn.0 += inPts[i].0; muIn.1 += inPts[i].1; muIn.2 += inPts[i].2
            muPr.0 += prPts[i].0; muPr.1 += prPts[i].1; muPr.2 += prPts[i].2
        }
        let fn = Float(N)
        muIn = (muIn.0/fn, muIn.1/fn, muIn.2/fn)
        muPr = (muPr.0/fn, muPr.1/fn, muPr.2/fn)

        // --- centred points & σ²_pred ----------------------------------------
        var sigma2: Float = 0
        var h = [Float](repeating: 0, count: 9)  // 3×3 cross-covariance (row-major)

        for i in 0..<N {
            let px = prPts[i].0 - muPr.0
            let py = prPts[i].1 - muPr.1
            let pz = prPts[i].2 - muPr.2
            let qx = inPts[i].0 - muIn.0
            let qy = inPts[i].1 - muIn.1
            let qz = inPts[i].2 - muIn.2
            sigma2 += px*px + py*py + pz*pz
            // H += q * p^T  (outer product, row-major)
            h[0] += qx*px; h[1] += qx*py; h[2] += qx*pz
            h[3] += qy*px; h[4] += qy*py; h[5] += qy*pz
            h[6] += qz*px; h[7] += qz*py; h[8] += qz*pz
        }
        sigma2 /= fn
        for j in 0..<9 { h[j] /= fn }

        guard sigma2 > 1e-12 else { return 1.0 }

        // --- SVD of H (3×3, float, LAPACK sgesdd) ----------------------------
        // sgesdd_ expects column-major; H is row-major ⇒ transpose.
        var at = [Float](repeating: 0, count: 9)
        for r in 0..<3 { for c in 0..<3 { at[c*3+r] = h[r*3+c] } }

        var svdM: __CLPK_integer = 3
        var svdN: __CLPK_integer = 3
        var lda: __CLPK_integer = 3
        var s = [Float](repeating: 0, count: 3)
        var u = [Float](repeating: 0, count: 9)
        var ldu: __CLPK_integer = 3
        var vt = [Float](repeating: 0, count: 9)
        var ldvt: __CLPK_integer = 3
        var work = [Float](repeating: 0, count: 128)
        var lwork: __CLPK_integer = 128
        var iwork = [__CLPK_integer](repeating: 0, count: 24)
        var info: __CLPK_integer = 0
        var jobz: Int8 = Int8(UnicodeScalar("A").value)

        sgesdd_(
            &jobz,
            &svdM, &svdN, &at, &lda, &s, &u, &ldu, &vt, &ldvt,
            &work, &lwork, &iwork, &info
        )
        guard info == 0 else { return 1.0 }  // SVD failed — fall back

        // det(U * V^T):  U and Vt are column-major from LAPACK.
        // U_col_major → U row-major: u_rm[r][c] = u[c*3+r]
        // V^T col-major → V^T row-major: vt_rm[r][c] = vt[c*3+r]
        // product M = U * V^T  (in row-major)
        func rm(_ buf: [Float], _ r: Int, _ c: Int) -> Float { buf[c*3+r] }
        var prod = [Float](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                var v: Float = 0
                for k in 0..<3 { v += rm(u, r, k) * rm(vt, k, c) }
                prod[r*3+c] = v
            }
        }
        let det = prod[0]*(prod[4]*prod[8]-prod[5]*prod[7])
                - prod[1]*(prod[3]*prod[8]-prod[5]*prod[6])
                + prod[2]*(prod[3]*prod[7]-prod[4]*prod[6])
        let d: Float = det < 0 ? -1 : 1

        // s = (s[0] + s[1] + d * s[2]) / σ²_pred
        let trSD = s[0] + s[1] + d * s[2]
        let scale = trSD / sigma2

        return scale > 1e-8 ? scale : 1.0
    }

    // MARK: - Point-cloud unprojection --------------------------------------

    /// Unproject a single depth map to world-space coloured points.
    static func unprojectToWorld(
        depth: [Float],
        confidence: [Float],
        confThreshold: Float,
        extrinsic4x4: [Float],
        intrinsic3x3: [Float],
        image: CGImage,
        W: Int, H: Int
    ) -> [ColoredPoint] {
        // c2w = inv(w2c).  For a rigid transform [R|t], c2w = [R^T | -R^T t ; 0 0 0 1].
        let c2w = invertRigid4x4(extrinsic4x4)

        let fx = intrinsic3x3[0]
        let fy = intrinsic3x3[4]
        let cx = intrinsic3x3[2]
        let cy = intrinsic3x3[5]

        // Read pixel colours from the image (resize to W×H).
        let colors = readImageColors(image, W: W, H: H)

        var pts = [ColoredPoint]()
        pts.reserveCapacity(W * H / 4)  // heuristic reservation

        for y in 0..<H {
            for x in 0..<W {
                let idx = y * W + x
                let d = depth[idx]
                if d <= 0 { continue }
                if confidence[idx] < confThreshold { continue }

                // Camera-space point.
                let camX = (Float(x) - cx) / fx * d
                let camY = (Float(y) - cy) / fy * d
                let camZ = d

                // World-space: p_world = c2w * [camX, camY, camZ, 1]
                let wx = c2w[0]*camX + c2w[1]*camY + c2w[2]*camZ  + c2w[3]
                let wy = c2w[4]*camX + c2w[5]*camY + c2w[6]*camZ  + c2w[7]
                let wz = c2w[8]*camX + c2w[9]*camY + c2w[10]*camZ + c2w[11]

                let ci = min(idx, colors.count - 1)
                pts.append(ColoredPoint(
                    position: (wx, wy, wz),
                    color: colors[ci]
                ))
            }
        }
        return pts
    }

    // MARK: - Geometry helpers ----------------------------------------------

    /// Invert a rigid 4×4 transform stored as 16 floats (row-major).
    static func invertRigid4x4(_ m: [Float]) -> [Float] {
        // R^T
        let r00 = m[0], r01 = m[4], r02 = m[8]
        let r10 = m[1], r11 = m[5], r12 = m[9]
        let r20 = m[2], r21 = m[6], r22 = m[10]
        let tx = m[3], ty = m[7], tz = m[11]
        // -R^T * t
        let itx = -(r00*tx + r01*ty + r02*tz)
        let ity = -(r10*tx + r11*ty + r12*tz)
        let itz = -(r20*tx + r21*ty + r22*tz)
        return [
            r00, r10, r20, itx,
            r01, r11, r21, ity,
            r02, r12, r22, itz,
            0,   0,   0,   1
        ]
    }

    /// Read the pixel colours of a CGImage, resized to W×H.  Returns an
    /// array of length W*H with (r,g,b) tuples in 0-1 range (row-major).
    static func readImageColors(_ image: CGImage, W: Int, H: Int) -> [(r: Float, g: Float, b: Float)] {
        var buf = [UInt8](repeating: 0, count: W * H * 4)
        guard let ctx = CGContext(
            data: &buf, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            return Array(repeating: (0,0,0), count: W * H)
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: W, height: H))

        var colors = [(r: Float, g: Float, b: Float)]()
        colors.reserveCapacity(W * H)
        for i in stride(from: 0, to: buf.count, by: 4) {
            colors.append((Float(buf[i]) / 255, Float(buf[i+1]) / 255, Float(buf[i+2]) / 255))
        }
        return colors
    }

    /// Compute the p-th percentile of a Float array (0..100).
    static func percentile(_ arr: [Float], p: Float) -> Float {
        guard !arr.isEmpty, p > 0 else { return -Float.infinity }
        if p >= 100 { return Float.infinity }
        var sorted = arr
        sorted.sort()
        let idx = max(0, min(sorted.count - 1, Int(Float(sorted.count) * p / 100)))
        return sorted[idx]
    }
}

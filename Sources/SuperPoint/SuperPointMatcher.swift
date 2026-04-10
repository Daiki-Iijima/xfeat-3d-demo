import CoreML
import Accelerate
import UIKit

// MARK: - Public types

struct SuperPointFeatures: @unchecked Sendable {
    /// Keypoints in processing-image coordinate space (procSize).
    let keypoints: [SIMD2<Float>]
    /// Flat row-major array, `count × 256`, each 256-element row is L2-normalised.
    let descriptors: [Float]
    var count: Int { keypoints.count }
}

// MARK: - SuperPointMatcher

/// Runs SuperPoint CoreML inference and mutual-nearest-neighbour matching.
/// Model outputs:
///   "semi" – (1, 65, H/8, W/8) detector logits + dustbin channel
///   "desc" – (1,256, H/8, W/8) dense unit-norm descriptors
final class SuperPointMatcher: @unchecked Sendable {

    static let shared = SuperPointMatcher()

    // Must match convert_superpoint_to_coreml.py PROC_H / PROC_W
    let procSize = CGSize(width: 960, height: 720)
    private let cell    = 8    // encoder stride
    private let descDim = 256

    var topK:       Int   = 1000 { didSet { invalidateCache() } }
    var nmsRadius:  Int   = 4    { didSet { invalidateCache() } }  // in cell-grid units
    var minCossim:  Float = 0.75
    var isVertical: Bool  = false

    private var model: MLModel?
    private var cachedRef: (image: UIImage, features: SuperPointFeatures)?

    func invalidateCache() { cachedRef = nil }

    private init() { loadModel() }

    // MARK: - Public

    func match(reference: UIImage, query: UIImage) -> MatchOutput? {
        let refFeats: SuperPointFeatures
        if let cached = cachedRef, cached.image === reference {
            refFeats = cached.features
        } else {
            guard let f = extractFeatures(from: reference) else { return nil }
            refFeats = f
            cachedRef = (reference, f)
        }
        guard let queryFeats = extractFeatures(from: query) else { return nil }

        let pairs = matchFeatures(ref: refFeats, query: queryFeats)
        let image = drawMatches(
            leftImage:  reference,
            rightImage: query,
            leftKP:     pairs.map { $0.0 },
            rightKP:    pairs.map { $0.1 }
        )
        return MatchOutput(image: image, matchCount: pairs.count)
    }

    // MARK: - Feature extraction

    func extractFeatures(from image: UIImage) -> SuperPointFeatures? {
        guard let model else { return nil }

        let w = Int(procSize.width)
        let h = Int(procSize.height)

        guard let gray = image.toGrayscaleFloat(width: w, height: h) else { return nil }

        // ── Build input ───────────────────────────────────────────────────
        guard let mlInput = try? MLMultiArray(
            shape: [1, 1, h as NSNumber, w as NSNumber],
            dataType: .float32
        ) else { return nil }

        let inputPtr = mlInput.dataPointer.assumingMemoryBound(to: Float.self)
        gray.withUnsafeBufferPointer { buf in
            inputPtr.assign(from: buf.baseAddress!, count: h * w)
        }

        // ── Inference ────────────────────────────────────────────────────
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": mlInput]),
              let output   = try? model.prediction(from: provider),
              let semiMLA  = output.featureValue(for: "semi")?.multiArrayValue,
              let descMLA  = output.featureValue(for: "desc")?.multiArrayValue
        else { return nil }

        guard semiMLA.shape.count == 4, descMLA.shape.count == 4 else { return nil }

        let fH = Int(truncating: semiMLA.shape[2])
        let fW = Int(truncating: semiMLA.shape[3])

        // Dense copy via MLShapedArray (handles non-contiguous strides safely)
        let semiFlat = MLShapedArray<Float>(converting: semiMLA).scalars  // [65 * fH * fW]
        let descFlat = MLShapedArray<Float>(converting: descMLA).scalars  // [256 * fH * fW]

        // ── Extract keypoints ─────────────────────────────────────────────
        let keypoints = extractKeypoints(semiFlat: semiFlat, fH: fH, fW: fW)
        guard !keypoints.isEmpty else {
            return SuperPointFeatures(keypoints: [], descriptors: [])
        }

        // ── Sample descriptors (keypoints are in proc-image coords) ───────
        let scaleX = Float(fW) / Float(w)
        let scaleY = Float(fH) / Float(h)
        let fmKP   = keypoints.map { SIMD2<Float>($0.x * scaleX, $0.y * scaleY) }

        let descriptors = sampleDescriptors(descFlat: descFlat, keypoints: fmKP, fH: fH, fW: fW)

        return SuperPointFeatures(keypoints: keypoints, descriptors: descriptors)
    }

    // MARK: - Keypoint extraction

    /// Extracts keypoints from the 65-channel semi tensor.
    ///
    /// Algorithm:
    ///  1. For each cell (cy, cx): softmax over 65 channels (64 grid + 1 dustbin)
    ///  2. Best probability across the 64 non-dustbin channels → cell score
    ///  3. Argmax channel → sub-cell pixel offset (dy = c / 8, dx = c % 8)
    ///  4. Cell-grid NMS: keep cells that are local maxima within nmsRadius cells
    ///  5. Return top-K keypoints in proc-image coordinates
    private func extractKeypoints(semiFlat: [Float], fH: Int, fW: Int) -> [SIMD2<Float>] {
        let numC = 65
        var cellScores = [Float](repeating: 0, count: fH * fW)
        var cellBestC  = [Int](repeating: 0, count: fH * fW)

        // Reusable scratch buffers to avoid per-cell heap allocations
        var logits = [Float](repeating: 0, count: numC)
        var exps   = [Float](repeating: 0, count: numC)

        for cy in 0..<fH {
            for cx in 0..<fW {
                // Gather logits for all 65 channels at this cell
                for c in 0..<numC {
                    logits[c] = semiFlat[c * fH * fW + cy * fW + cx]
                }
                // Stable softmax
                var maxL: Float = logits[0]
                for c in 1..<numC { if logits[c] > maxL { maxL = logits[c] } }
                var expSum: Float = 0
                for c in 0..<numC {
                    let e = expf(logits[c] - maxL)
                    exps[c] = e
                    expSum += e
                }
                let invSum = 1.0 / expSum
                // Best non-dustbin channel (c < 64)
                var best: Float = -1, bestC = 0
                for c in 0..<64 {
                    let p = exps[c] * invSum
                    if p > best { best = p; bestC = c }
                }
                cellScores[cy * fW + cx] = best
                cellBestC[cy * fW + cx]  = bestC
            }
        }

        // Cell-grid NMS
        let r         = max(1, nmsRadius)
        let threshold = Float(0.015)
        var peaks: [(SIMD2<Float>, Float)] = []

        for cy in r..<(fH - r) {
            for cx in r..<(fW - r) {
                let s = cellScores[cy * fW + cx]
                guard s > threshold else { continue }
                var isMax = true
                outer: for dy in -r...r {
                    for dx in -r...r {
                        if dy == 0 && dx == 0 { continue }
                        if cellScores[(cy + dy) * fW + (cx + dx)] >= s {
                            isMax = false; break outer
                        }
                    }
                }
                if isMax {
                    let c  = cellBestC[cy * fW + cx]
                    let dy = c / cell
                    let dx = c % cell
                    let px = Float(cx * cell + dx)
                    let py = Float(cy * cell + dy)
                    peaks.append((SIMD2<Float>(px, py), s))
                }
            }
        }

        return peaks.sorted { $0.1 > $1.1 }.prefix(topK).map { $0.0 }
    }

    // MARK: - Descriptor sampling

    /// Bilinear-samples the dense descriptor map at keypoint locations and L2-normalises.
    /// `keypoints` are in feature-map coordinates (fH × fW space).
    private func sampleDescriptors(
        descFlat: [Float],
        keypoints: [SIMD2<Float>],
        fH: Int, fW: Int
    ) -> [Float] {
        let C = descDim
        let N = keypoints.count
        var descriptors = [Float](repeating: 0, count: N * C)

        descFlat.withUnsafeBufferPointer { fp in
            descriptors.withUnsafeMutableBufferPointer { buf in
                for (ki, kp) in keypoints.enumerated() {
                    let x = kp.x, y = kp.y
                    let x0 = max(0, min(fW - 1, Int(x)))
                    let y0 = max(0, min(fH - 1, Int(y)))
                    let x1 = min(fW - 1, x0 + 1)
                    let y1 = min(fH - 1, y0 + 1)
                    let wx = x - Float(x0)
                    let wy = y - Float(y0)
                    let w00 = (1 - wy) * (1 - wx)
                    let w01 = (1 - wy) * wx
                    let w10 = wy * (1 - wx)
                    let w11 = wy * wx

                    let dstBase = buf.baseAddress! + ki * C

                    for c in 0..<C {
                        let base = c * fH * fW
                        let v00  = fp[base + y0 * fW + x0]
                        let v01  = fp[base + y0 * fW + x1]
                        let v10  = fp[base + y1 * fW + x0]
                        let v11  = fp[base + y1 * fW + x1]
                        dstBase[c] = w00*v00 + w01*v01 + w10*v10 + w11*v11
                    }

                    // L2 normalise (bilinear interp breaks unit-norm)
                    var norm: Float = 0
                    vDSP_svesq(dstBase, 1, &norm, vDSP_Length(C))
                    norm = max(sqrtf(norm), 1e-8)
                    var invNorm = 1.0 / norm
                    vDSP_vsmul(dstBase, 1, &invNorm, dstBase, 1, vDSP_Length(C))
                }
            }
        }

        return descriptors
    }

    // MARK: - Mutual NN matching

    private func matchFeatures(
        ref: SuperPointFeatures,
        query: SuperPointFeatures
    ) -> [(SIMD2<Float>, SIMD2<Float>)] {
        let N = ref.count, M = query.count, C = descDim
        guard N > 0, M > 0 else { return [] }

        // cosSimMatrix (N×M) = refDesc @ queryDescᵀ via BLAS sgemm
        var cosSimMatrix = [Float](repeating: 0, count: N * M)
        ref.descriptors.withUnsafeBufferPointer { rp in
            query.descriptors.withUnsafeBufferPointer { qp in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(N), Int32(M), Int32(C),
                    1.0, rp.baseAddress!, Int32(C),
                         qp.baseAddress!, Int32(C),
                    0.0, &cosSimMatrix,   Int32(M)
                )
            }
        }

        // Forward matches (ref → query)
        var match10 = [Int](repeating: 0, count: N)
        for i in 0..<N {
            var best: Float = -2, bestJ = 0
            for j in 0..<M {
                let v = cosSimMatrix[i * M + j]
                if v > best { best = v; bestJ = j }
            }
            match10[i] = bestJ
        }

        // Backward matches (query → ref)
        var match01 = [Int](repeating: 0, count: M)
        for j in 0..<M {
            var best: Float = -2, bestI = 0
            for i in 0..<N {
                let v = cosSimMatrix[i * M + j]
                if v > best { best = v; bestI = i }
            }
            match01[j] = bestI
        }

        // Mutual check + cosine threshold
        var pairs: [(SIMD2<Float>, SIMD2<Float>)] = []
        for i in 0..<N {
            let j = match10[i]
            guard cosSimMatrix[i * M + j] >= minCossim, match01[j] == i else { continue }
            pairs.append((ref.keypoints[i], query.keypoints[j]))
        }
        return pairs
    }

    // MARK: - Drawing

    private func drawMatches(
        leftImage:  UIImage, rightImage: UIImage,
        leftKP: [SIMD2<Float>], rightKP: [SIMD2<Float>]
    ) -> UIImage {
        isVertical
            ? drawMatchesVertical(leftImage: leftImage, rightImage: rightImage,
                                  leftKP: leftKP, rightKP: rightKP)
            : drawMatchesHorizontal(leftImage: leftImage, rightImage: rightImage,
                                    leftKP: leftKP, rightKP: rightKP)
    }

    private func drawMatchesHorizontal(
        leftImage: UIImage, rightImage: UIImage,
        leftKP: [SIMD2<Float>], rightKP: [SIMD2<Float>]
    ) -> UIImage {
        let dispH: CGFloat = 480
        let lW = leftImage.size.width  * (dispH / leftImage.size.height)
        let rW = rightImage.size.width * (dispH / rightImage.size.height)
        let pW = Float(procSize.width), pH = Float(procSize.height)

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: lW + rW, height: dispH))
        return renderer.image { ctx in
            let cg = ctx.cgContext
            UIColor.black.setFill()
            cg.fill(CGRect(x: 0, y: 0, width: lW + rW, height: dispH))
            leftImage.draw(in:  CGRect(x: 0,  y: 0, width: lW, height: dispH))
            rightImage.draw(in: CGRect(x: lW, y: 0, width: rW, height: dispH))
            guard !leftKP.isEmpty else { return }
            let lSx = lW / CGFloat(pW), lSy = dispH / CGFloat(pH)
            let rSx = rW / CGFloat(pW), rSy = dispH / CGFloat(pH)
            cg.setStrokeColor(UIColor.yellow.withAlphaComponent(0.7).cgColor)
            cg.setLineWidth(1.0)
            for (l, r) in zip(leftKP, rightKP) {
                cg.move(to:    CGPoint(x: CGFloat(l.x)*lSx,      y: CGFloat(l.y)*lSy))
                cg.addLine(to: CGPoint(x: CGFloat(r.x)*rSx + lW, y: CGFloat(r.y)*rSy))
            }
            cg.strokePath()
            cg.setFillColor(UIColor.orange.cgColor)
            for l in leftKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(l.x)*lSx - 2,      y: CGFloat(l.y)*lSy - 2, width: 4, height: 4))
            }
            for r in rightKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(r.x)*rSx + lW - 2, y: CGFloat(r.y)*rSy - 2, width: 4, height: 4))
            }
        }
    }

    private func drawMatchesVertical(
        leftImage: UIImage, rightImage: UIImage,
        leftKP: [SIMD2<Float>], rightKP: [SIMD2<Float>]
    ) -> UIImage {
        let dispW: CGFloat = 480
        let lH = leftImage.size.height  * (dispW / leftImage.size.width)
        let rH = rightImage.size.height * (dispW / rightImage.size.width)
        let pW = Float(procSize.width), pH = Float(procSize.height)

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: dispW, height: lH + rH))
        return renderer.image { ctx in
            let cg = ctx.cgContext
            UIColor.black.setFill()
            cg.fill(CGRect(x: 0, y: 0, width: dispW, height: lH + rH))
            leftImage.draw(in:  CGRect(x: 0, y: 0,  width: dispW, height: lH))
            rightImage.draw(in: CGRect(x: 0, y: lH, width: dispW, height: rH))
            guard !leftKP.isEmpty else { return }
            let lSx = dispW / CGFloat(pW), lSy = lH / CGFloat(pH)
            let rSx = dispW / CGFloat(pW), rSy = rH / CGFloat(pH)
            cg.setStrokeColor(UIColor.yellow.withAlphaComponent(0.7).cgColor)
            cg.setLineWidth(1.0)
            for (l, r) in zip(leftKP, rightKP) {
                cg.move(to:    CGPoint(x: CGFloat(l.x)*lSx, y: CGFloat(l.y)*lSy))
                cg.addLine(to: CGPoint(x: CGFloat(r.x)*rSx, y: CGFloat(r.y)*rSy + lH))
            }
            cg.strokePath()
            cg.setFillColor(UIColor.orange.cgColor)
            for l in leftKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(l.x)*lSx - 2, y: CGFloat(l.y)*lSy - 2,      width: 4, height: 4))
            }
            for r in rightKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(r.x)*rSx - 2, y: CGFloat(r.y)*rSy + lH - 2, width: 4, height: 4))
            }
        }
    }

    // MARK: - Model loading

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "SuperPoint", withExtension: "mlmodelc") else {
            print("[SuperPointMatcher] SuperPoint.mlmodelc not found in bundle.")
            return
        }
        do {
            let cfg = MLModelConfiguration()
            // Use cpuOnly to avoid GPU/ANE padding issues (same reason as XFeatMatcher)
            cfg.computeUnits = .cpuOnly
            model = try MLModel(contentsOf: url, configuration: cfg)
            print("[SuperPointMatcher] Model loaded.")
        } catch {
            print("[SuperPointMatcher] Load error: \(error)")
        }
    }
}

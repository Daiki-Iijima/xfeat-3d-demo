import CoreML
import Accelerate
import UIKit

// MARK: - ALIKEDMatcher

/// Runs ALIKED CoreML inference (dense backbone only, no SDDH) and
/// XFeat-style mutual-nearest-neighbour matching.
///
/// The CoreML model outputs a dense feat_map + score_map; NMS and
/// bilinear descriptor sampling are done here in Swift — the same
/// pipeline as XFeatMatcher.
final class ALIKEDMatcher: @unchecked Sendable {

    static let shared = ALIKEDMatcher()

    // Must match PROC_H / PROC_W in convert_aliked_to_coreml.py
    let procSize = CGSize(width: 960, height: 720)

    var topK:       Int   = 1000 { didSet { invalidateCache() } }
    var nmsRadius:  Int   = 2    { didSet { invalidateCache() } }
    var minCossim:  Float = 0.82
    var isVertical: Bool  = false

    private var model: MLModel?
    private var cachedRef: (image: UIImage, features: ALIKEDFeatures)?

    func invalidateCache() { cachedRef = nil }

    private init() { loadModel() }

    // MARK: - Public

    func match(reference: UIImage, query: UIImage) -> MatchOutput? {
        let refFeats: ALIKEDFeatures
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

    func extractFeatures(from image: UIImage) -> ALIKEDFeatures? {
        guard let model else { return nil }

        let w = Int(procSize.width)
        let h = Int(procSize.height)

        // RGB float [0,1] — ImageNet normalisation is baked into the CoreML model
        guard let rgb = image.toRGBFloat(width: w, height: h) else { return nil }

        // Build input MLMultiArray (1, 3, H, W)
        guard let mlInput = try? MLMultiArray(
            shape: [1, 3, h as NSNumber, w as NSNumber],
            dataType: .float32
        ) else { return nil }

        let inputPtr = mlInput.dataPointer.assumingMemoryBound(to: Float.self)
        rgb.withUnsafeBufferPointer { buf in
            inputPtr.assign(from: buf.baseAddress!, count: 3 * h * w)
        }

        // Inference
        guard let provider = try? MLDictionaryFeatureProvider(dictionary: ["image": mlInput]),
              let output      = try? model.prediction(from: provider),
              let featsMLA    = output.featureValue(for: "feats")?.multiArrayValue,
              let heatmapMLA  = output.featureValue(for: "heatmap")?.multiArrayValue
        else { return nil }

        guard featsMLA.shape.count == 4,
              heatmapMLA.shape.count == 4 else { return nil }

        let fH = Int(truncating: featsMLA.shape[2])
        let fW = Int(truncating: featsMLA.shape[3])
        let C  = Int(truncating: featsMLA.shape[1])

        // Dense copy via MLShapedArray (handles GPU padding strides)
        let featsFlat   = MLShapedArray<Float>(converting: featsMLA).scalars
        let heatmapFlat = MLShapedArray<Float>(converting: heatmapMLA).scalars

        // NMS
        let keypoints = nms(heatmapFlat: heatmapFlat, fH: fH, fW: fW)
        guard !keypoints.isEmpty else {
            return ALIKEDFeatures(keypoints: [], descriptors: [], descDim: C)
        }

        // Bilinear sample + L2 normalise
        var descriptors = [Float](repeating: 0, count: keypoints.count * C)

        featsFlat.withUnsafeBufferPointer { fp in
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
                        let v00 = fp[base + y0 * fW + x0]
                        let v01 = fp[base + y0 * fW + x1]
                        let v10 = fp[base + y1 * fW + x0]
                        let v11 = fp[base + y1 * fW + x1]
                        dstBase[c] = w00*v00 + w01*v01 + w10*v10 + w11*v11
                    }

                    var norm: Float = 0
                    vDSP_svesq(dstBase, 1, &norm, vDSP_Length(C))
                    norm = max(sqrtf(norm), 1e-8)
                    var invNorm = 1.0 / norm
                    vDSP_vsmul(dstBase, 1, &invNorm, dstBase, 1, vDSP_Length(C))
                }
            }
        }

        let scaleX = Float(w) / Float(fW)
        let scaleY = Float(h) / Float(fH)
        let imgKP = keypoints.map { SIMD2<Float>($0.x * scaleX, $0.y * scaleY) }

        return ALIKEDFeatures(keypoints: imgKP, descriptors: descriptors, descDim: C)
    }

    // MARK: - NMS

    private func nms(heatmapFlat: [Float], fH: Int, fW: Int) -> [SIMD2<Float>] {
        // heatmapFlat is already in [0,1] — sigmoid is baked into the CoreML model.
        let scores = heatmapFlat

        let r = nmsRadius
        var peaks: [(SIMD2<Float>, Float)] = []

        for y in r..<(fH - r) {
            for x in r..<(fW - r) {
                let s = scores[y * fW + x]
                guard s > 0.01 else { continue }
                var isMax = true
                outer: for dy in -r...r {
                    for dx in -r...r {
                        if dy == 0 && dx == 0 { continue }
                        if scores[(y + dy) * fW + (x + dx)] >= s {
                            isMax = false
                            break outer
                        }
                    }
                }
                if isMax { peaks.append((SIMD2<Float>(Float(x), Float(y)), s)) }
            }
        }

        return peaks.sorted { $0.1 > $1.1 }.prefix(topK).map { $0.0 }
    }

    // MARK: - Mutual NN matching

    private func matchFeatures(
        ref: ALIKEDFeatures,
        query: ALIKEDFeatures
    ) -> [(SIMD2<Float>, SIMD2<Float>)] {
        let N = ref.count, M = query.count, C = ref.descDim
        guard N > 0, M > 0, C > 0 else { return [] }

        var cosSimMatrix = [Float](repeating: 0, count: N * M)
        ref.descriptors.withUnsafeBufferPointer { rp in
            query.descriptors.withUnsafeBufferPointer { qp in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(N), Int32(M), Int32(C),
                    1.0, rp.baseAddress!, Int32(C),
                         qp.baseAddress!, Int32(C),
                    0.0, &cosSimMatrix, Int32(M)
                )
            }
        }

        var match10 = [Int](repeating: 0, count: N)
        for i in 0..<N {
            var best: Float = -2, bestJ = 0
            for j in 0..<M {
                let v = cosSimMatrix[i * M + j]
                if v > best { best = v; bestJ = j }
            }
            match10[i] = bestJ
        }

        var match01 = [Int](repeating: 0, count: M)
        for j in 0..<M {
            var best: Float = -2, bestI = 0
            for i in 0..<N {
                let v = cosSimMatrix[i * M + j]
                if v > best { best = v; bestI = i }
            }
            match01[j] = bestI
        }

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
        leftImage:  UIImage,
        rightImage: UIImage,
        leftKP:  [SIMD2<Float>],
        rightKP: [SIMD2<Float>]
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
            cg.setStrokeColor(UIColor.green.withAlphaComponent(0.7).cgColor)
            cg.setLineWidth(1.0)
            for (l, r) in zip(leftKP, rightKP) {
                cg.move(to:    CGPoint(x: CGFloat(l.x)*lSx,      y: CGFloat(l.y)*lSy))
                cg.addLine(to: CGPoint(x: CGFloat(r.x)*rSx + lW, y: CGFloat(r.y)*rSy))
            }
            cg.strokePath()
            cg.setFillColor(UIColor.red.cgColor)
            for l in leftKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(l.x)*lSx - 2, y: CGFloat(l.y)*lSy - 2,
                                          width: 4, height: 4))
            }
            for r in rightKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(r.x)*rSx + lW - 2, y: CGFloat(r.y)*rSy - 2,
                                          width: 4, height: 4))
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
            cg.setStrokeColor(UIColor.green.withAlphaComponent(0.7).cgColor)
            cg.setLineWidth(1.0)
            for (l, r) in zip(leftKP, rightKP) {
                cg.move(to:    CGPoint(x: CGFloat(l.x)*lSx, y: CGFloat(l.y)*lSy))
                cg.addLine(to: CGPoint(x: CGFloat(r.x)*rSx, y: CGFloat(r.y)*rSy + lH))
            }
            cg.strokePath()
            cg.setFillColor(UIColor.red.cgColor)
            for l in leftKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(l.x)*lSx - 2, y: CGFloat(l.y)*lSy - 2,
                                          width: 4, height: 4))
            }
            for r in rightKP {
                cg.fillEllipse(in: CGRect(x: CGFloat(r.x)*rSx - 2, y: CGFloat(r.y)*rSy + lH - 2,
                                          width: 4, height: 4))
            }
        }
    }

    // MARK: - Model loading

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "ALIKED", withExtension: "mlmodelc") else {
            print("[ALIKEDMatcher] ALIKED.mlmodelc not found in bundle.")
            return
        }
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuOnly   // avoid GPU padding stride issues
            model = try MLModel(contentsOf: url, configuration: cfg)
            print("[ALIKEDMatcher] Model loaded.")
        } catch {
            print("[ALIKEDMatcher] Load error: \(error)")
        }
    }
}

// MARK: - ALIKEDFeatures

struct ALIKEDFeatures: @unchecked Sendable {
    let keypoints:   [SIMD2<Float>]
    let descriptors: [Float]   // row-major, each row L2-normalised
    let descDim:     Int
    var count: Int { keypoints.count }
}

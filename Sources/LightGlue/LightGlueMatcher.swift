import CoreML
import Accelerate
import UIKit

/// LightGlue neural feature matcher using SIFT descriptors extracted via OpenCV.
///
/// Requires `LightGlue.mlpackage` compiled into the app bundle.
/// When the model is absent, falls back to mutual nearest-neighbour matching.
///
/// Use `matchKeypoints(reference:query:)` to obtain raw 2D correspondence pairs
/// for downstream PnP localization.
final class LightGlueMatcher: @unchecked Sendable {

    static let shared = LightGlueMatcher()

    // ── Parameters ────────────────────────────────────────────────────────
    var siftTopK:  Int   = 512
    var procWidth: Int   = 960
    var minScore:  Float = 0.2

    // ── Internals ─────────────────────────────────────────────────────────
    // Must match convert_lightglue_to_coreml.py MAX_KP
    private let maxKP = 512
    private var model: MLModel?
    private var cachedRef: (image: UIImage, features: ExtractedSIFTFeatures)?

    private init() { loadModel() }

    func invalidateCache() { cachedRef = nil }

    // MARK: - Public

    /// Returns raw matched keypoint pairs (ref space, query space).
    /// Reference and query keypoints are in their respective procSize coordinate spaces.
    func matchKeypoints(reference: UIImage, query: UIImage)
        -> [(refKP: SIMD2<Float>, queryKP: SIMD2<Float>)]?
    {
        guard let (pairs, _, _) = computePairs(reference: reference, query: query) else { return nil }
        return pairs.map { (refKP: $0.0, queryKP: $0.1) }
    }

    // MARK: - Private: shared feature extraction + matching

    private func computePairs(reference: UIImage, query: UIImage)
        -> ([(SIMD2<Float>, SIMD2<Float>)], ExtractedSIFTFeatures, ExtractedSIFTFeatures)?
    {
        let refFeats: ExtractedSIFTFeatures
        if let cached = cachedRef, cached.image === reference {
            refFeats = cached.features
        } else {
            guard let f = extractFeatures(from: reference) else { return nil }
            refFeats = f
            cachedRef = (reference, f)
        }
        guard let queryFeats = extractFeatures(from: query) else { return nil }

        let pairs: [(SIMD2<Float>, SIMD2<Float>)]
        if model != nil {
            pairs = runLightGlue(ref: refFeats, query: queryFeats)
        } else {
            pairs = mutualNN(ref: refFeats, query: queryFeats)
        }
        return (pairs, refFeats, queryFeats)
    }

    // MARK: - SIFT extraction via OpenCV

    func extractFeatures(from image: UIImage) -> ExtractedSIFTFeatures? {
        guard let raw = OpenCVBridge.extractSIFT(
            from: image, topK: Int32(siftTopK), procWidth: Int32(procWidth)
        ), raw.count > 0 else { return nil }

        let count   = Int(raw.count)
        let descDim = 128

        let keypoints: [SIMD2<Float>] = raw.keypointsXY.withUnsafeBytes { ptr in
            let f = ptr.bindMemory(to: Float.self)
            return (0..<count).map { SIMD2<Float>(f[$0 * 2], f[$0 * 2 + 1]) }
        }
        let descriptors: [Float] = raw.descriptors.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self).prefix(count * descDim))
        }
        let scales: [Float] = raw.scales.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self).prefix(count))
        }
        let orientations: [Float] = raw.orientations.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self).prefix(count))
        }

        return ExtractedSIFTFeatures(
            keypoints:    keypoints,
            descriptors:  descriptors,
            descDim:      descDim,
            procSize:     raw.procSize,
            scales:       scales,
            orientations: orientations
        )
    }

    // MARK: - LightGlue CoreML inference

    private func runLightGlue(
        ref:   ExtractedSIFTFeatures,
        query: ExtractedSIFTFeatures
    ) -> [(SIMD2<Float>, SIMD2<Float>)] {
        guard let model else { return [] }

        let N0 = min(ref.count,   maxKP)
        let N1 = min(query.count, maxKP)
        guard N0 > 0, N1 > 0 else { return [] }

        let D = ref.descDim  // 128

        guard
            let kpts0MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber, 2], dataType: .float32),
            let scales0MLA = try? MLMultiArray(shape: [1, maxKP as NSNumber],    dataType: .float32),
            let oris0MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber],    dataType: .float32),
            let desc0MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber, D as NSNumber], dataType: .float32),
            let kpts1MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber, 2], dataType: .float32),
            let scales1MLA = try? MLMultiArray(shape: [1, maxKP as NSNumber],    dataType: .float32),
            let oris1MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber],    dataType: .float32),
            let desc1MLA   = try? MLMultiArray(shape: [1, maxKP as NSNumber, D as NSNumber], dataType: .float32)
        else { return [] }

        fillKeypoints(ref.keypoints,   count: N0, procSize: ref.procSize,   into: kpts0MLA)
        fillKeypoints(query.keypoints, count: N1, procSize: query.procSize, into: kpts1MLA)
        fillScalars(ref.scales,         count: N0, into: scales0MLA)
        fillScalars(query.scales,       count: N1, into: scales1MLA)
        fillScalars(ref.orientations,   count: N0, into: oris0MLA)
        fillScalars(query.orientations, count: N1, into: oris1MLA)
        fillDescriptors(ref.descriptors,   count: N0, dim: D, into: desc0MLA)
        fillDescriptors(query.descriptors, count: N1, dim: D, into: desc1MLA)

        guard
            let provider = try? MLDictionaryFeatureProvider(dictionary: [
                "kpts0": kpts0MLA, "scales0": scales0MLA, "oris0": oris0MLA, "desc0": desc0MLA,
                "kpts1": kpts1MLA, "scales1": scales1MLA, "oris1": oris1MLA, "desc1": desc1MLA,
            ]),
            let out       = try? model.prediction(from: provider),
            let m0MLA     = out.featureValue(for: "matches0")?.multiArrayValue,
            let scoresMLA = out.featureValue(for: "scores")?.multiArrayValue
        else { return [] }

        let matchesFlat = MLShapedArray<Float>(converting: m0MLA).scalars
        let scoresFlat  = MLShapedArray<Float>(converting: scoresMLA).scalars

        var pairs: [(SIMD2<Float>, SIMD2<Float>)] = []
        for i in 0..<N0 {
            let j = Int(matchesFlat[i])
            guard j >= 0, j < N1 else { continue }
            guard scoresFlat[i] >= minScore else { continue }
            pairs.append((ref.keypoints[i], query.keypoints[j]))
        }
        return pairs
    }

    // MARK: - Fill helpers

    private func fillKeypoints(
        _ kpts: [SIMD2<Float>], count: Int, procSize: CGSize,
        into mla: MLMultiArray
    ) {
        let w     = Float(procSize.width)
        let h     = Float(procSize.height)
        let shift = SIMD2<Float>(w / 2, h / 2)
        let scale = max(w, h) / 2
        let ptr   = mla.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            let nkp    = (kpts[i] - shift) / scale
            ptr[i * 2]     = nkp.x
            ptr[i * 2 + 1] = nkp.y
        }
    }

    private func fillScalars(_ values: [Float], count: Int, into mla: MLMultiArray) {
        guard !values.isEmpty else { return }
        let ptr = mla.dataPointer.assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { bp in
            ptr.assign(from: bp.baseAddress!, count: min(count, values.count))
        }
    }

    private func fillDescriptors(_ descs: [Float], count: Int, dim: Int, into mla: MLMultiArray) {
        let ptr = mla.dataPointer.assumingMemoryBound(to: Float.self)
        descs.withUnsafeBufferPointer { bp in
            ptr.assign(from: bp.baseAddress!, count: count * dim)
        }
    }

    // MARK: - Mutual NN fallback (used when model is not loaded)

    private func mutualNN(
        ref:   ExtractedSIFTFeatures,
        query: ExtractedSIFTFeatures
    ) -> [(SIMD2<Float>, SIMD2<Float>)] {
        let N = ref.count, M = query.count, D = ref.descDim
        guard N > 0, M > 0 else { return [] }

        var sim = [Float](repeating: 0, count: N * M)
        ref.descriptors.withUnsafeBufferPointer { rp in
            query.descriptors.withUnsafeBufferPointer { qp in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(N), Int32(M), Int32(D),
                    1.0, rp.baseAddress!, Int32(D),
                         qp.baseAddress!, Int32(D),
                    0.0, &sim,           Int32(M)
                )
            }
        }

        var match10 = [Int](repeating: 0, count: N)
        for i in 0..<N {
            var best: Float = -2; var bestJ = 0
            for j in 0..<M { if sim[i * M + j] > best { best = sim[i * M + j]; bestJ = j } }
            match10[i] = bestJ
        }
        var match01 = [Int](repeating: 0, count: M)
        for j in 0..<M {
            var best: Float = -2; var bestI = 0
            for i in 0..<N { if sim[i * M + j] > best { best = sim[i * M + j]; bestI = i } }
            match01[j] = bestI
        }

        var pairs: [(SIMD2<Float>, SIMD2<Float>)] = []
        for i in 0..<N {
            let j = match10[i]
            guard sim[i * M + j] >= minScore, match01[j] == i else { continue }
            pairs.append((ref.keypoints[i], query.keypoints[j]))
        }
        return pairs
    }

    // MARK: - Model loading

    private func loadModel() {
        guard let url = Bundle.main.url(forResource: "LightGlue", withExtension: "mlmodelc") else {
            print("[LightGlueMatcher] LightGlue.mlmodelc not in bundle — using mutual-NN fallback.")
            return
        }
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndNeuralEngine
            model = try MLModel(contentsOf: url, configuration: cfg)
            print("[LightGlueMatcher] Model loaded.")
        } catch {
            print("[LightGlueMatcher] Load error: \(error)")
        }
    }
}

// MARK: - SIFT Feature Container

struct ExtractedSIFTFeatures {
    let keypoints:    [SIMD2<Float>]
    let descriptors:  [Float]
    let descDim:      Int
    let procSize:     CGSize
    let scales:       [Float]
    let orientations: [Float]

    var count: Int { keypoints.count }
}

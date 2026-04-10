import Foundation
import UIKit
import simd
import Accelerate

/// Tracks feature points across frames using LK optical flow for propagation
/// and one of several extractor strategies for (re-)detection.
///
/// **Hierarchical mode** (`.hierarchical`):
/// Three async pipelines fire at different intervals and build a quality pyramid:
/// - ORB   – every `orbInterval` frames   → fast, broad coverage, position-only
/// - XFeat – every `xfeatInterval` frames → medium quality, 64-dim descriptor
/// - ALIKED– every `alikedInterval` frames → highest quality, 128-dim descriptor
///
/// When a higher-tier detector finds a point near / matching a lower-tier track,
/// it *promotes* that track in-place (keeps ID, updates descriptor + tier).
/// This gives every point a chance to become more reliable over time.
///
/// Threading: `update(frame:)` is not concurrent-safe; call it from one thread at a time.
final class PointTracker: @unchecked Sendable {

    static let shared = PointTracker()

    // MARK: - Configuration

    var targetCount: Int = 500
    var minActiveCount: Int = 200
    var redetectInterval: Int = 60   // single-extractor mode
    var procWidth:  Int = 960
    var procHeight: Int = 720
    var extractor: TrackingExtractor = .hierarchical

    // Hierarchical-mode intervals
    var orbInterval:    Int = 10
    var xfeatInterval:  Int = 30
    var alikedInterval: Int = 45

    // Per-tier merge/suppression radius (pixels in procSize space).
    // A new detection is considered "already tracked" if an existing point
    // of the same or higher tier lies within this radius.
    var orbMergeRadius:    Float = 15
    var xfeatMergeRadius:  Float = 15
    var alikedMergeRadius: Float = 15

    // Per-tier point budget (hierarchical mode only).
    // Total active points = min(orbBudget + xfeatBudget + alikedBudget, targetCount).
    var orbBudget:    Int = 150
    var xfeatBudget:  Int = 150
    var alikedBudget: Int = 300  // increased: more ALIKED pts → more reliable Essential Matrix

    /// Highest tier that will be computed. Tiers above this are never scheduled.
    /// Setting this to `.orb` disables XFeat and ALIKED entirely (cheapest mode).
    var maxTier: TrackTier = .aliked

    var procSize: CGSize { CGSize(width: procWidth, height: procHeight) }

    // MARK: - Internal types

    private struct PendingDetection: Sendable {
        let keypoints:   [SIMD2<Float>]
        let descriptors: [Float]   // empty for ORB
        let descDim:     Int       // 0 for ORB
        let tier:        TrackTier
    }

    // MARK: - State (all accesses protected by `lock`)

    private let lock = NSLock()
    private var trackedPoints: [TrackedPoint] = []
    private var nextID: Int = 0
    private var prevGrayData: NSData?
    private var frameCount: Int = 0

    // Single-extractor
    private var isDetecting = false
    private var pendingDetection: PendingDetection?

    // Hierarchical
    private var isDetectingORB    = false
    private var isDetectingXFeat  = false
    private var isDetectingALIKED = false
    private var pendingORB:    PendingDetection?
    private var pendingXFeat:  PendingDetection?
    private var pendingALIKED: PendingDetection?

    private init() {}

    // MARK: - Public API

    func update(frame: UIImage) -> [TrackedPoint] {
        lock.lock()
        defer { lock.unlock() }

        // 1. Apply pending detections.
        if extractor == .hierarchical {
            // Apply highest-tier first so promotions happen before adding ORB fills.
            if let p = pendingALIKED { pendingALIKED = nil; mergePending(p) }
            if let p = pendingXFeat  { pendingXFeat  = nil; mergePending(p) }
            if let p = pendingORB    { pendingORB    = nil; mergePending(p) }
        } else {
            if let p = pendingDetection { pendingDetection = nil; mergePending(p) }
        }

        // 2. Grayscale at proc resolution.
        let currGray = OpenCVBridge.toGray(
            frame, width: Int32(procWidth), height: Int32(procHeight)
        ) as NSData

        // 3. LK optical flow — preserve IDs, descriptors, tiers.
        if let prev = prevGrayData, !trackedPoints.isEmpty {
            let positions = trackedPoints.map(\.position)
            let ptsData   = packPoints(positions)
            let result    = OpenCVBridge.trackLK(
                ptsData as Data,
                count:    trackedPoints.count,
                prevGray: prev as Data,
                currGray: currGray as Data,
                width:    Int32(procWidth),
                height:   Int32(procHeight)
            )
            let statusPtr = result.status.withUnsafeBytes   { $0.bindMemory(to: UInt8.self) }
            let ptsPtr    = result.pointsXY.withUnsafeBytes { $0.bindMemory(to: Float.self) }
            let fw = Float(procWidth), fh = Float(procHeight)

            var active: [TrackedPoint] = []
            active.reserveCapacity(trackedPoints.count)
            for i in 0..<trackedPoints.count {
                guard statusPtr[i] == 1 else { continue }
                let x = ptsPtr[i * 2], y = ptsPtr[i * 2 + 1]
                guard x >= 0, x < fw, y >= 0, y < fh else { continue }
                let old = trackedPoints[i]
                active.append(TrackedPoint(
                    id: old.id, position: SIMD2(x, y),
                    age: old.age + 1,
                    descriptor: old.descriptor, tier: old.tier
                ))
            }
            trackedPoints = active
        }

        // Drop tracks above the current maxTier (e.g. if user switched ORB→XFeat only)
        if extractor == .hierarchical {
            trackedPoints = trackedPoints.filter { $0.tier <= maxTier }
        }

        prevGrayData = currGray
        frameCount  += 1

        // 4. Schedule async re-detections.
        if extractor == .hierarchical {
            scheduleHierarchical(frame: frame)
        } else {
            scheduleSingle(frame: frame)
        }

        return trackedPoints
    }

    func reset() {
        lock.lock()
        defer { lock.unlock() }
        trackedPoints    = []
        nextID           = 0
        prevGrayData     = nil
        frameCount       = 0
        isDetecting      = false
        isDetectingORB   = false
        isDetectingXFeat = false
        isDetectingALIKED = false
        pendingDetection = nil
        pendingORB    = nil
        pendingXFeat  = nil
        pendingALIKED = nil
    }

    // MARK: - Scheduling (called with lock held)

    private func scheduleSingle(frame: UIImage) {
        let needDetect = trackedPoints.count < minActiveCount
            || frameCount % redetectInterval == 0
        guard needDetect, !isDetecting else { return }
        isDetecting = true
        let capturedFrame = frame
        let target = targetCount
        let ext = extractor
        Task.detached(priority: .background) { [weak self] in
            guard let self else { return }
            let det = self.detectSingle(frame: capturedFrame, target: target, extractor: ext)
            self.lock.lock()
            if let det { self.pendingDetection = det }
            self.isDetecting = false
            self.lock.unlock()
        }
    }

    private func scheduleHierarchical(frame: UIImage) {
        let target = targetCount
        let pw = procWidth, ph = procHeight

        // ORB: most frequent — fills gaps quickly.
        if (frameCount % orbInterval == 0 || trackedPoints.count < minActiveCount), !isDetectingORB {
            isDetectingORB = true
            let f = frame
            Task.detached(priority: .background) { [weak self] in
                guard let self else { return }
                let det = self.detectORB(frame: f, topK: target, procWidth: pw, procHeight: ph)
                self.lock.lock()
                if let det { self.pendingORB = det }
                self.isDetectingORB = false
                self.lock.unlock()
            }
        }

        // XFeat: medium interval — adds descriptor-backed tracks.
        if maxTier >= .xfeat, frameCount % xfeatInterval == 0, !isDetectingXFeat {
            isDetectingXFeat = true
            let f = frame
            Task.detached(priority: .background) { [weak self] in
                guard let self else { return }
                let det = self.detectXFeat(frame: f, topK: target)
                self.lock.lock()
                if let det { self.pendingXFeat = det }
                self.isDetectingXFeat = false
                self.lock.unlock()
            }
        }

        // ALIKED: slowest — highest quality, promotes existing tracks.
        if maxTier >= .aliked, frameCount % alikedInterval == 0, !isDetectingALIKED {
            isDetectingALIKED = true
            let f = frame
            Task.detached(priority: .background) { [weak self] in
                guard let self else { return }
                let det = self.detectALIKED(frame: f, topK: target)
                self.lock.lock()
                if let det { self.pendingALIKED = det }
                self.isDetectingALIKED = false
                self.lock.unlock()
            }
        }
    }

    // MARK: - Detectors (called off main thread, no lock needed)

    private func detectSingle(frame: UIImage, target: Int, extractor: TrackingExtractor) -> PendingDetection? {
        switch extractor {
        case .xfeat:
            let saved = XFeatMatcher.shared.topK
            XFeatMatcher.shared.topK = target
            let f = XFeatMatcher.shared.extractFeatures(from: frame)
            XFeatMatcher.shared.topK = saved
            guard let f, !f.keypoints.isEmpty else { return nil }
            let dim = f.descriptors.count / f.keypoints.count
            return PendingDetection(keypoints: f.keypoints, descriptors: f.descriptors, descDim: dim, tier: .xfeat)

        case .aliked:
            let saved = ALIKEDMatcher.shared.topK
            ALIKEDMatcher.shared.topK = target
            let f = ALIKEDMatcher.shared.extractFeatures(from: frame)
            ALIKEDMatcher.shared.topK = saved
            guard let f, !f.keypoints.isEmpty else { return nil }
            return PendingDetection(keypoints: f.keypoints, descriptors: f.descriptors, descDim: f.descDim, tier: .aliked)

        case .superpoint:
            let saved = SuperPointMatcher.shared.topK
            SuperPointMatcher.shared.topK = target
            let f = SuperPointMatcher.shared.extractFeatures(from: frame)
            SuperPointMatcher.shared.topK = saved
            guard let f, !f.keypoints.isEmpty else { return nil }
            let dim = f.descriptors.count / f.keypoints.count
            return PendingDetection(keypoints: f.keypoints, descriptors: f.descriptors, descDim: dim, tier: .xfeat)

        case .hierarchical:
            return nil  // handled separately
        }
    }

    private func detectORB(frame: UIImage, topK: Int, procWidth: Int, procHeight: Int) -> PendingDetection? {
        guard let result = OpenCVBridge.detectORB(
            from: frame, topK: Int32(topK),
            procWidth: Int32(procWidth), procHeight: Int32(procHeight)
        ) else { return nil }
        guard result.count > 0 else { return nil }
        let ptr = result.keypointsXY.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        var kps = [SIMD2<Float>](repeating: .zero, count: result.count)
        for i in 0..<result.count { kps[i] = SIMD2(ptr[i * 2], ptr[i * 2 + 1]) }
        return PendingDetection(keypoints: kps, descriptors: [], descDim: 0, tier: .orb)
    }

    private func detectXFeat(frame: UIImage, topK: Int) -> PendingDetection? {
        let saved = XFeatMatcher.shared.topK
        XFeatMatcher.shared.topK = topK
        let f = XFeatMatcher.shared.extractFeatures(from: frame)
        XFeatMatcher.shared.topK = saved
        guard let f, !f.keypoints.isEmpty else { return nil }
        let dim = f.descriptors.count / f.keypoints.count
        return PendingDetection(keypoints: f.keypoints, descriptors: f.descriptors, descDim: dim, tier: .xfeat)
    }

    private func detectALIKED(frame: UIImage, topK: Int) -> PendingDetection? {
        let saved = ALIKEDMatcher.shared.topK
        ALIKEDMatcher.shared.topK = topK
        let f = ALIKEDMatcher.shared.extractFeatures(from: frame)
        ALIKEDMatcher.shared.topK = saved
        guard let f, !f.keypoints.isEmpty else { return nil }
        return PendingDetection(keypoints: f.keypoints, descriptors: f.descriptors, descDim: f.descDim, tier: .aliked)
    }

    // MARK: - Descriptor-based merge (called with lock held)

    /// Merge a batch of newly-detected keypoints into the current track set.
    ///
    /// - For detections **with** descriptors (XFeat / ALIKED):
    ///   1. Run mutual-NN matching against existing tracks that have descriptors.
    ///   2. Matched existing tracks are *promoted* (tier ↑, descriptor refreshed; ID & position unchanged).
    ///   3. Also check proximity against ORB-tier tracks (no descriptor) — promote those too.
    ///   4. Truly new detections get a fresh ID.
    /// - For ORB detections (no descriptor):
    ///   Proximity suppression only — add if no track is within 15 px.
    private func mergePending(_ detection: PendingDetection) {
        let newKPs   = detection.keypoints
        let newDescs = detection.descriptors
        let D        = detection.descDim
        let tier     = detection.tier
        let M        = newKPs.count
        let N        = trackedPoints.count

        guard M > 0 else { return }

        var matchedNew = Set<Int>()  // new-detection indices already accounted for

        // ── Descriptor matching ─────────────────────────────────────────────
        if D > 0 && N > 0 {
            // Eligible existing tracks: those whose descriptor dimension matches.
            let eligible = trackedPoints.indices.filter { trackedPoints[$0].descriptor.count == D }

            if !eligible.isEmpty {
                let E = eligible.count
                let existingDescs = eligible.flatMap { trackedPoints[$0].descriptor }

                // S[i, j] = cosine similarity between existing[i] and new[j]
                var S = [Float](repeating: 0, count: E * M)
                existingDescs.withUnsafeBufferPointer { ep in
                    newDescs.withUnsafeBufferPointer { np in
                        cblas_sgemm(
                            CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(E), Int32(M), Int32(D),
                            1.0, ep.baseAddress!, Int32(D),
                                 np.baseAddress!, Int32(D),
                            0.0, &S, Int32(M)
                        )
                    }
                }

                // Mutual nearest-neighbour
                var nn_E2N = [Int](repeating: -1, count: E)
                for i in 0..<E {
                    var best: Float = -1; var bestJ = -1
                    for j in 0..<M { if S[i * M + j] > best { best = S[i * M + j]; bestJ = j } }
                    nn_E2N[i] = bestJ
                }
                var nn_N2E = [Int](repeating: -1, count: M)
                for j in 0..<M {
                    var best: Float = -1; var bestI = -1
                    for i in 0..<E { if S[i * M + j] > best { best = S[i * M + j]; bestI = i } }
                    nn_N2E[j] = bestI
                }

                let threshold: Float = 0.7
                for j in 0..<M {
                    let i = nn_N2E[j]
                    guard i >= 0, nn_E2N[i] == j, S[i * M + j] >= threshold else { continue }
                    // Promote: keep ID + position (LK is authoritative), refresh descriptor + tier.
                    let globalIdx = eligible[i]
                    let old = trackedPoints[globalIdx]
                    let fresh = Array(newDescs[(j * D)..<((j + 1) * D)])
                    trackedPoints[globalIdx] = TrackedPoint(
                        id: old.id, position: old.position, age: old.age,
                        descriptor: fresh,
                        tier: max(old.tier, tier)  // never demote
                    )
                    matchedNew.insert(j)
                }
            }

            // Also try proximity-promote ORB-tier tracks (they have no descriptor).
            if tier > .orb {
                let promoRadius: Float = tier == .xfeat ? xfeatMergeRadius : alikedMergeRadius
                for j in 0..<M where !matchedNew.contains(j) {
                    let newPos = newKPs[j]
                    if let orbIdx = trackedPoints.firstIndex(where: {
                        $0.tier == .orb && simd_length($0.position - newPos) < promoRadius
                    }) {
                        let fresh = Array(newDescs[(j * D)..<((j + 1) * D)])
                        let old = trackedPoints[orbIdx]
                        trackedPoints[orbIdx] = TrackedPoint(
                            id: old.id, position: old.position, age: old.age,
                            descriptor: fresh, tier: tier
                        )
                        matchedNew.insert(j)
                    }
                }
            }
        } else {
            // ORB: suppress if any existing track is within orbMergeRadius.
            for j in 0..<M {
                if trackedPoints.contains(where: {
                    simd_length($0.position - newKPs[j]) < orbMergeRadius
                }) {
                    matchedNew.insert(j)
                }
            }
        }

        // ── Add unmatched detections as new tracks ──────────────────────────
        var merged = trackedPoints
        for j in 0..<M where !matchedNew.contains(j) {
            let desc = D > 0 ? Array(newDescs[(j * D)..<((j + 1) * D)]) : []
            merged.append(TrackedPoint(
                id: nextID, position: newKPs[j], age: 0, descriptor: desc, tier: tier
            ))
            nextID += 1
        }

        // ── Cap per-tier then overall ────────────────────────────────────────
        if extractor == .hierarchical {
            var alikedPts = merged.filter { $0.tier == .aliked }.sorted { $0.age > $1.age }
            var xfeatPts  = merged.filter { $0.tier == .xfeat  }.sorted { $0.age > $1.age }
            var orbPts    = merged.filter { $0.tier == .orb    }.sorted { $0.age > $1.age }
            if alikedPts.count > alikedBudget { alikedPts = Array(alikedPts.prefix(alikedBudget)) }
            if xfeatPts.count  > xfeatBudget  { xfeatPts  = Array(xfeatPts.prefix(xfeatBudget))  }
            if orbPts.count    > orbBudget     { orbPts    = Array(orbPts.prefix(orbBudget))     }
            merged = alikedPts + xfeatPts + orbPts
        }
        if merged.count > targetCount {
            // Within same tier, older tracks have priority.
            merged.sort { lhs, rhs in
                if lhs.tier != rhs.tier { return lhs.tier > rhs.tier }
                return lhs.age > rhs.age
            }
            merged = Array(merged.prefix(targetCount))
        }
        trackedPoints = merged
    }

    // MARK: - Helpers

    private func packPoints(_ points: [SIMD2<Float>]) -> NSData {
        var buf = [Float](repeating: 0, count: points.count * 2)
        for (i, p) in points.enumerated() { buf[i * 2] = p.x; buf[i * 2 + 1] = p.y }
        return NSData(bytes: buf, length: buf.count * MemoryLayout<Float>.size)
    }
}

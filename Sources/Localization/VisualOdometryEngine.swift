import Foundation
import UIKit
import simd
import Accelerate
import CoreGraphics

/// Marker-initialized Visual Odometry engine.
///
/// ## Algorithm
///
/// 1. **Searching**: Detect ArUco marker every frame to establish the world coordinate frame.
/// 2. **Bootstrapping**: Store the initial pose; wait for the camera to move enough
///    (sufficient parallax), then triangulate the first set of 3D map points.
/// 3. **Tracking**: Match ALIKED descriptors against the visual map → solvePnP → camera pose.
///    Every `triInterval` frames, triangulate new points from the current and reference keyframe
///    to grow the map.
/// 4. **Lost**: PnP fails → try ArUco re-detection; on success, resume tracking without clearing
///    the existing map.
///
/// The world coordinate origin is the detected ArUco marker.
/// All 3D positions are in metres when ArUco-ArUco bootstrap is used.
@MainActor
final class VisualOdometryEngine: ObservableObject {

    // MARK: - State Machine

    enum Phase: Equatable, CustomStringConvertible {
        case searching
        case bootstrapping
        case tracking
        case lost

        var description: String {
            switch self {
            case .searching:     return "ArUco検索中"
            case .bootstrapping: return "初期化中"
            case .tracking:      return "トラッキング中"
            case .lost:          return "ロスト"
            }
        }
    }

    @Published private(set) var phase: Phase = .searching
    /// Camera-to-world transform (world = marker coordinate frame).
    @Published private(set) var cameraPose: simd_float4x4?
    @Published private(set) var mapPointCount: Int = 0
    @Published private(set) var pnpInlierCount: Int = 0
    @Published private(set) var matchCount: Int = 0
    @Published private(set) var statusMessage: String = "ArUcoマーカーをかざしてください"

    /// IDs of tracked points that were matched to the visual map and used in the last PnP solve.
    @Published private(set) var lastMatchedIDs: Set<Int> = []
    /// IDs of tracked points that were just triangulated into new map points.
    @Published private(set) var lastTriangulatedIDs: Set<Int> = []
    /// True if the ArUco marker was detected within the last ~10 frames.
    @Published private(set) var isMarkerInView: Bool = false

    private var lastMarkerFrame: Int = -999

    let pointCloud = PointCloud()
    private let visualMap = VisualMap()

    /// Descriptor dimension currently in use (64 for XFeat, 128 for ALIKED, 0 = none).
    /// If this changes, the visual map is automatically cleared and tracking restarts.
    private var activeDescDim: Int = 0

    // MARK: - Tunable Parameters

    /// Physical size of the ArUco marker in metres.
    var markerSizeMeters: Float = 0.15
    /// Minimum average pixel displacement needed to trigger bootstrap triangulation.
    var minBootstrapParallax: Float = 20.0
    /// Minimum number of common ALIKED tracks between bootstrap and current frame.
    var minBootstrapCommon: Int = 12
    /// Minimum PnP RANSAC inliers to accept the estimated pose.
    var minPnPInliers: Int = 8
    /// Frames between triangulation attempts in tracking mode.
    var triInterval: Int = 20
    /// Minimum parallax (px) needed for tracking-phase triangulation.
    var minTriParallax: Float = 8.0
    /// Minimum common ALIKED tracks needed for tracking-phase triangulation.
    var minTriCommon: Int = 10
    /// Maximum reprojection error (px) to accept a triangulated point.
    var maxReprojError: Float = 2.5

    let procW: Float = 960
    let procH: Float = 720

    // MARK: - Bootstrap State

    private var bootstrapPose: simd_float4x4?
    /// Flat L2-normalised descriptor matrix for the bootstrap frame (N × 128).
    private var bootstrapDescriptors: [Float] = []
    /// 2D positions of bootstrap ALIKED points (parallel to bootstrapDescriptors rows).
    private var bootstrapPositions2D: [SIMD2<Float>] = []

    // MARK: - Triangulation Reference Keyframe

    private var triRefPose: simd_float4x4?
    private var triRefPositions: [Int: SIMD2<Float>] = [:]

    private var frameIndex: Int = 0
    private var lastTriFrame: Int = -999

    // MARK: - Public API

    func reset() {
        phase = .searching
        cameraPose = nil
        mapPointCount = 0
        pnpInlierCount = 0
        matchCount = 0
        lastMatchedIDs = []
        lastTriangulatedIDs = []
        isMarkerInView = false
        lastMarkerFrame = -999
        activeDescDim = 0
        visualMap.clear()
        pointCloud.clear()
        bootstrapPose = nil
        bootstrapDescriptors = []
        bootstrapPositions2D = []
        triRefPose = nil
        triRefPositions = [:]
        frameIndex = 0
        lastTriFrame = -999
        statusMessage = "ArUcoマーカーをかざしてください"
    }

    /// Process one camera frame.
    ///
    /// - Parameters:
    ///   - procImage: Frame resized to procW × procH (used for ArUco detection and color sampling).
    ///   - trackedPoints: All tracked points from PointTracker (mixed tiers).
    ///   - intrinsics: Camera intrinsics at proc resolution (fx, fy, cx, cy).
    func processFrame(
        procImage: UIImage,
        trackedPoints: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) {
        frameIndex += 1

        // Pick the highest tier that has descriptors (ALIKED 128-dim → XFeat 64-dim → none)
        let topPts = Self.topDescriptorPoints(trackedPoints)
        let descDim = topPts.first?.descriptor.count ?? 0

        if descDim == 0 {
            statusMessage = "ORBのみ — 記述子なし。XFeat か ALIKED を有効にしてください"
            return
        }

        // If the descriptor dimension changed (user switched tier), reset the map
        if activeDescDim != 0 && descDim != activeDescDim {
            visualMap.clear(); pointCloud.clear()
            mapPointCount = 0; lastTriangulatedIDs = []; lastMatchedIDs = []
            bootstrapDescriptors = []; bootstrapPositions2D = []
            phase = .searching
            statusMessage = "ティア変更 — マップをリセット"
            activeDescDim = descDim
            return
        }
        activeDescDim = descDim

        switch phase {
        case .searching, .lost:
            tryMarkerInit(procImage: procImage, aliked: topPts, intrinsics: intrinsics, descDim: descDim)
        case .bootstrapping:
            tryBootstrap(aliked: topPts, intrinsics: intrinsics, procImage: procImage, descDim: descDim)
        case .tracking:
            trackingStep(aliked: topPts, intrinsics: intrinsics, procImage: procImage, descDim: descDim)
        }
    }

    /// Returns points from the highest available tier that has descriptors.
    private static func topDescriptorPoints(_ points: [TrackedPoint]) -> [TrackedPoint] {
        for tier in [TrackTier.aliked, .xfeat] {
            let pts = points.filter { $0.tier == tier && !$0.descriptor.isEmpty }
            if !pts.isEmpty { return pts }
        }
        return []
    }

    // MARK: - Phase: Searching / Lost

    private func tryMarkerInit(
        procImage: UIImage,
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        descDim: Int
    ) {
        let aruco = detectArUco(procImage: procImage, intrinsics: intrinsics)
        guard let aruco else {
            statusMessage = phase == .lost
                ? "ロスト中: マーカーを再検出中..."
                : "ArUcoマーカーをかざしてください"
            return
        }

        // Always update the bootstrap reference to the latest marker frame.
        // ALIKED points may not be ready yet right after detection, so we keep
        // refreshing until we have enough, then transition to bootstrapping.
        let n = aliked.count
        var descs = [Float](repeating: 0, count: n * descDim)
        for (i, pt) in aliked.enumerated() {
            var norm2: Float = 0
            vDSP_svesq(pt.descriptor, 1, &norm2, vDSP_Length(descDim))
            let scale = 1.0 / max(sqrtf(norm2), 1e-8)
            vDSP_vsmul(pt.descriptor, 1, [scale], &descs[i * descDim], 1, vDSP_Length(descDim))
        }
        bootstrapPose = aruco
        bootstrapDescriptors = descs
        bootstrapPositions2D = aliked.map { SIMD2<Float>($0.position.x, $0.position.y) }
        cameraPose = aruco

        lastMarkerFrame = frameIndex
        isMarkerInView  = true
        print("[VO] MarkerInit: aliked=\(n) boot=\(bootstrapPositions2D.count) need=\(minBootstrapCommon)")

        if n >= minBootstrapCommon {
            phase = .bootstrapping
            statusMessage = "マーカー検出! カメラをゆっくり動かしてください (boot:\(n)点)"
        } else {
            // Stay in searching — keep updating reference until ALIKED points are ready
            statusMessage = "マーカー検出済み — ALIKED待機中 \(n)/\(minBootstrapCommon)点"
        }
    }

    // MARK: - Phase: Bootstrapping

    private func tryBootstrap(
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        descDim: Int
    ) {
        guard let bPose = bootstrapPose else { phase = .searching; return }

        let bootN = bootstrapPositions2D.count
        let curN  = aliked.count

        // If bootstrap frame had no ALIKED points (detector not ready), fall back to searching
        guard bootN >= minBootstrapCommon else {
            print("[VO] Bootstrap: boot=\(bootN) too few — back to searching")
            phase = .searching
            return
        }

        // Descriptor-based matching (ID-independent — works after PointTracker re-detects)
        let (pts1, pts2, common) = matchBootstrapDescriptors(current: aliked, descDim: descDim)

        print("[VO] Bootstrap: boot=\(bootN) cur=\(curN) matched=\(common.count) need=\(minBootstrapCommon)")

        guard common.count >= minBootstrapCommon else {
            statusMessage = "共通点不足 \(common.count)/\(minBootstrapCommon) [boot:\(bootN) cur:\(curN)]"
            return
        }

        let parallax = averageDisplacement(pts1: pts1, pts2: pts2, count: common.count)
        guard parallax >= minBootstrapParallax else {
            statusMessage = String(format: "視差: %.1fpx / 必要: %.0fpx — もう少し動かしてください",
                                   parallax, minBootstrapParallax)
            return
        }

        // Determine current pose
        // Prefer ArUco (metric scale) → fall back to Essential Matrix
        let currentPose: simd_float4x4
        if let arucoNow = detectArUco(procImage: procImage, intrinsics: intrinsics) {
            currentPose = arucoNow
        } else {
            guard let em = essentialMatrixPose(
                pose1: bPose, pts1: pts1, pts2: pts2, count: common.count, intrinsics: intrinsics
            ) else {
                statusMessage = "姿勢推定失敗 — マーカーを視野内に保ちながら移動してください"
                return
            }
            currentPose = em
        }

        // Triangulate initial map points
        let result = triangulate(
            pose1: bPose, pose2: currentPose,
            pts1: pts1, pts2: pts2,
            aliked: common,
            intrinsics: intrinsics,
            procImage: procImage,
            confidence: 0.5,
            descDim: descDim
        )

        guard result.mapEntries.count >= 8 else {
            statusMessage = "三角測量点が少なすぎます (\(result.mapEntries.count)) — もっと移動してください"
            return
        }

        visualMap.add(result.mapEntries)
        pointCloud.add(result.cloud3D)
        mapPointCount = visualMap.count
        lastTriangulatedIDs = Set(result.triangulatedIDs)

        cameraPose = currentPose
        triRefPose = currentPose
        triRefPositions = Dictionary(uniqueKeysWithValues: aliked.map { ($0.id, $0.position) })
        lastTriFrame = frameIndex

        phase = .tracking
        statusMessage = "初期化完了! マップ: \(mapPointCount)点"
    }

    // MARK: - Phase: Tracking

    private func trackingStep(
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        descDim: Int
    ) {
        guard !visualMap.isEmpty else { phase = .bootstrapping; return }

        // Update marker visibility (fades out after ~10 frames without detection)
        isMarkerInView = (frameIndex - lastMarkerFrame) <= 10

        let (fx, fy, cx, cy) = intrinsics
        let n = aliked.count
        guard n >= 6 else { statusMessage = "ALIKED点が不足しています"; return }

        // Build flat query descriptor matrix for visual-map matching
        var queryDescs = [Float](repeating: 0, count: n * descDim)
        for (i, pt) in aliked.enumerated() {
            let base = i * descDim
            for j in 0..<descDim { queryDescs[base + j] = pt.descriptor[j] }
        }

        let matches = visualMap.findMatches(
            queryDescriptors: queryDescs,
            queryCount: n,
            descDim: descDim,
            minScore: 0.70
        )
        matchCount = matches.count
        // Record which tracked-point IDs are currently matched to the map
        lastMatchedIDs = Set(matches.map { aliked[$0.queryIdx].id })

        guard matches.count >= 6 else {
            print("[VO] Track: match=\(matches.count) aliked=\(n) map=\(visualMap.count) — lost")
            statusMessage = "マッチ不足 (\(matches.count)/6) — ロスト回復中"
            tryArUcoRecovery(procImage: procImage, intrinsics: intrinsics)
            return
        }

        // Build 3D-2D correspondence arrays for PnP
        var pts3D = [Float]()
        var pts2D = [Float]()
        pts3D.reserveCapacity(matches.count * 3)
        pts2D.reserveCapacity(matches.count * 2)

        for m in matches {
            let mp = visualMap.entries[m.entryIdx]
            let pt = aliked[m.queryIdx]
            pts3D.append(mp.position.x); pts3D.append(mp.position.y); pts3D.append(mp.position.z)
            pts2D.append(pt.position.x); pts2D.append(pt.position.y)
        }

        let pnp = OpenCVBridge.solvePnP(
            points3D: Data(bytes: pts3D, count: pts3D.count * 4),
            points2D: Data(bytes: pts2D, count: pts2D.count * 4),
            count: matches.count,
            fx: fx, fy: fy, cx: cx, cy: cy
        )

        guard pnp.success, pnp.inlierCount >= minPnPInliers else {
            print("[VO] PnP failed: inliers=\(pnp.inlierCount)/\(minPnPInliers) success=\(pnp.success)")
            statusMessage = "PnP失敗 (インライア: \(pnp.inlierCount)/\(minPnPInliers)) — 回復中"
            tryArUcoRecovery(procImage: procImage, intrinsics: intrinsics)
            return
        }

        let vm = buildViewMatrix(rData: pnp.rotationMatrix, tData: pnp.translationVector)
        let newPose = vm.inverse  // camera-to-world

        cameraPose = newPose
        pnpInlierCount = pnp.inlierCount
        statusMessage = "追跡中 — PnP: \(pnp.inlierCount)点 | マップ: \(mapPointCount)点 | マッチ: \(matchCount)点"

        // Periodically triangulate new map points to grow the map
        if frameIndex - lastTriFrame >= triInterval {
            growMap(currentPose: newPose, aliked: aliked,
                    intrinsics: intrinsics, procImage: procImage, descDim: descDim)
        }
    }

    // MARK: - Map Growth

    private func growMap(
        currentPose: simd_float4x4,
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        descDim: Int
    ) {
        guard let refPose = triRefPose else { return }

        var common: [TrackedPoint] = []
        var pts1 = [Float]()
        var pts2 = [Float]()

        for pt in aliked {
            guard let refPos = triRefPositions[pt.id] else { continue }
            common.append(pt)
            pts1.append(refPos.x); pts1.append(refPos.y)
            pts2.append(pt.position.x); pts2.append(pt.position.y)
        }

        guard common.count >= minTriCommon else { return }

        let parallax = averageDisplacement(pts1: pts1, pts2: pts2, count: common.count)
        guard parallax >= minTriParallax else { return }

        let result = triangulate(
            pose1: refPose, pose2: currentPose,
            pts1: pts1, pts2: pts2,
            aliked: common,
            intrinsics: intrinsics,
            procImage: procImage,
            confidence: 0.6,
            descDim: descDim
        )

        print("[VO] GrowMap: common=\(common.count) parallax=\(String(format:"%.1f",parallax))px tri=\(result.mapEntries.count)")
        guard !result.mapEntries.isEmpty else { return }

        visualMap.add(result.mapEntries)
        pointCloud.add(result.cloud3D)
        mapPointCount = visualMap.count
        lastTriangulatedIDs = Set(result.triangulatedIDs)

        // Advance reference keyframe so baseline grows for next attempt
        triRefPose = currentPose
        triRefPositions = Dictionary(uniqueKeysWithValues: aliked.map { ($0.id, $0.position) })
        lastTriFrame = frameIndex
    }

    // MARK: - ArUco Recovery (Lost / Tracking)

    private func tryArUcoRecovery(
        procImage: UIImage,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) {
        if let pose = detectArUco(procImage: procImage, intrinsics: intrinsics) {
            lastMarkerFrame = frameIndex
            isMarkerInView  = true
            cameraPose = pose
            // Survived without resetting map — resume tracking next frame
        } else {
            phase = .lost
        }
    }

    // MARK: - Triangulation

    private struct TriResult {
        let mapEntries: [VisualMapEntry]
        let cloud3D: [Point3D]
        let triangulatedIDs: [Int]  // TrackedPoint IDs of successfully triangulated points
    }

    private func triangulate(
        pose1: simd_float4x4,
        pose2: simd_float4x4,
        pts1: [Float],
        pts2: [Float],
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        confidence: Float,
        descDim: Int
    ) -> TriResult {
        let (fx, fy, cx, cy) = intrinsics
        let count = aliked.count
        guard count > 0, pts1.count == count * 2, pts2.count == count * 2 else {
            return TriResult(mapEntries: [], cloud3D: [], triangulatedIDs: [])
        }

        let V1 = pose1.inverse  // world-to-camera
        let V2 = pose2.inverse

        let P1 = projectionMatrix(from: V1)
        let P2 = projectionMatrix(from: V2)

        // Normalize 2D points to image coordinates
        var n1 = [Float](repeating: 0, count: count * 2)
        var n2 = [Float](repeating: 0, count: count * 2)
        for i in 0..<count {
            n1[i*2]   = (pts1[i*2]   - cx) / fx
            n1[i*2+1] = (pts1[i*2+1] - cy) / fy
            n2[i*2]   = (pts2[i*2]   - cx) / fx
            n2[i*2+1] = (pts2[i*2+1] - cy) / fy
        }

        guard let raw = OpenCVBridge.triangulatePoints(
            Data(bytes: P1, count: 48), proj2: Data(bytes: P2, count: 48),
            pts1: Data(bytes: n1, count: n1.count * 4),
            pts2: Data(bytes: n2, count: n2.count * 4),
            count: count
        ) else { return TriResult(mapEntries: [], cloud3D: [], triangulatedIDs: []) }

        let worldPts = unpackFloat3(from: raw, count: count)

        var mapEntries: [VisualMapEntry] = []
        var cloud3D: [Point3D] = []
        var triangulatedIDs: [Int] = []

        for (i, wp) in worldPts.enumerated() {
            guard wp != .zero else { continue }

            // Cheirality + depth range check in both cameras
            let c1 = V1 * SIMD4<Float>(wp, 1)
            let c2 = V2 * SIMD4<Float>(wp, 1)
            guard c1.z > 0.01, c1.z < 30, c2.z > 0.01 else { continue }

            // Reprojection error in frame 1
            let u1 = (c1.x / c1.z) * fx + cx
            let v1 = (c1.y / c1.z) * fy + cy
            let e1 = sqrt(pow(u1 - pts1[i*2], 2) + pow(v1 - pts1[i*2+1], 2))
            guard e1 < maxReprojError else { continue }

            // Reprojection error in frame 2
            let u2 = (c2.x / c2.z) * fx + cx
            let v2 = (c2.y / c2.z) * fy + cy
            let e2 = sqrt(pow(u2 - pts2[i*2], 2) + pow(v2 - pts2[i*2+1], 2))
            guard e2 < maxReprojError else { continue }

            let desc = aliked[i].descriptor
            guard desc.count == descDim else { continue }

            let color = sampleColor(procImage, x: pts2[i*2], y: pts2[i*2+1])
            mapEntries.append(VisualMapEntry(position: wp, descriptor: desc))
            cloud3D.append(Point3D(position: wp, color: color, confidence: confidence))
            triangulatedIDs.append(aliked[i].id)
        }

        return TriResult(mapEntries: mapEntries, cloud3D: cloud3D, triangulatedIDs: triangulatedIDs)
    }

    // MARK: - Bootstrap Descriptor Matching

    /// Match current ALIKED points to bootstrap descriptors using cosine similarity (BLAS).
    /// Returns parallel arrays: pts1 (bootstrap 2D), pts2 (current 2D), and matched TrackedPoints.
    /// This is ID-independent so it survives PointTracker re-detections.
    private func matchBootstrapDescriptors(
        current: [TrackedPoint], descDim: Int
    ) -> (pts1: [Float], pts2: [Float], common: [TrackedPoint]) {
        let N = bootstrapPositions2D.count
        let M = current.count
        guard N > 0, M > 0, descDim > 0 else { return ([], [], []) }

        // Build L2-normalised current descriptor matrix (M × descDim)
        var curMat = [Float](repeating: 0, count: M * descDim)
        for (j, pt) in current.enumerated() {
            guard pt.descriptor.count == descDim else { continue }
            var norm2: Float = 0
            vDSP_svesq(pt.descriptor, 1, &norm2, vDSP_Length(descDim))
            let scale = 1.0 / max(sqrtf(norm2), 1e-8)
            vDSP_vsmul(pt.descriptor, 1, [scale], &curMat[j * descDim], 1, vDSP_Length(descDim))
        }

        // Cosine similarity matrix: simMat[j,i] = dot(cur[j], boot[i])  (M×N)
        var simMat = [Float](repeating: 0, count: M * N)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(M), Int32(N), Int32(descDim),
                    1.0, curMat, Int32(descDim),
                    bootstrapDescriptors, Int32(descDim),
                    0.0, &simMat, Int32(N))

        let threshold: Float = 0.72
        var pts1 = [Float](); var pts2 = [Float](); var matched = [TrackedPoint]()
        pts1.reserveCapacity(M * 2); pts2.reserveCapacity(M * 2); matched.reserveCapacity(M)

        // For each current point, find its nearest bootstrap point
        var usedBootstrap = Set<Int>()
        for j in 0..<M {
            var bestSim: Float = threshold - 0.001
            var bestI = -1
            for i in 0..<N {
                let s = simMat[j * N + i]
                if s > bestSim { bestSim = s; bestI = i }
            }
            guard bestI >= 0, !usedBootstrap.contains(bestI) else { continue }
            usedBootstrap.insert(bestI)
            let bp = bootstrapPositions2D[bestI]
            pts1.append(bp.x); pts1.append(bp.y)
            pts2.append(current[j].position.x); pts2.append(current[j].position.y)
            matched.append(current[j])
        }
        return (pts1, pts2, matched)
    }

    // MARK: - Helpers

    /// Detect ArUco 5×5 marker and return camera-to-world pose on success.
    private func detectArUco(
        procImage: UIImage,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) -> simd_float4x4? {
        let (fx, fy, cx, cy) = intrinsics
        let aruco = OpenCVBridge.detectArUco5x5(
            procImage, markerSize: markerSizeMeters,
            procWidth: Int32(procW), procHeight: Int32(procH),
            fx: fx, fy: fy, cx: cx, cy: cy
        )
        guard aruco.detected else { return nil }
        let vm = buildViewMatrix(rData: aruco.rotationMatrix, tData: aruco.translationVector)
        return vm.inverse
    }

    /// Compute the second-frame camera-to-world pose using the Essential Matrix.
    /// The translation is unit-normalized (unknown metric scale unless ArUco is used).
    private func essentialMatrixPose(
        pose1: simd_float4x4,
        pts1: [Float], pts2: [Float], count: Int,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) -> simd_float4x4? {
        let (fx, fy, cx, cy) = intrinsics
        let rp = OpenCVBridge.recoverPose(
            pts1: Data(bytes: pts1, count: pts1.count * 4),
            pts2: Data(bytes: pts2, count: pts2.count * 4),
            count: count,
            fx: fx, fy: fy, cx: cx, cy: cy
        )
        guard rp.success, rp.inlierCount >= 10 else { return nil }

        let V1 = pose1.inverse  // world-to-camera at frame 1
        let relR = mat3FromData(rp.rotationMatrix)
        let relT = vec3FromData(rp.translationVector)

        // V2 = relative_T * V1
        let R1 = simd_float3x3(V1.columns.0.xyz, V1.columns.1.xyz, V1.columns.2.xyz)
        let t1 = V1.columns.3.xyz
        let R2 = relR * R1
        let t2 = relR * t1 + relT

        let V2 = simd_float4x4(
            SIMD4<Float>(R2.columns.0, 0),
            SIMD4<Float>(R2.columns.1, 0),
            SIMD4<Float>(R2.columns.2, 0),
            SIMD4<Float>(t2, 1)
        )
        return V2.inverse
    }

    // MARK: - Geometry Utilities

    /// Build a 4×4 view matrix (world-to-camera) from OpenCV rvec/tvec data.
    private func buildViewMatrix(rData: Data, tData: Data) -> simd_float4x4 {
        guard rData.count >= 36, tData.count >= 12 else { return matrix_identity_float4x4 }
        let rf = rData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        let tf = tData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        let R = simd_float3x3(
            SIMD3<Float>(rf[0], rf[3], rf[6]),
            SIMD3<Float>(rf[1], rf[4], rf[7]),
            SIMD3<Float>(rf[2], rf[5], rf[8])
        )
        let t = SIMD3<Float>(tf[0], tf[1], tf[2])
        return simd_float4x4(
            SIMD4<Float>(R.columns.0, 0),
            SIMD4<Float>(R.columns.1, 0),
            SIMD4<Float>(R.columns.2, 0),
            SIMD4<Float>(t, 1)
        )
    }

    /// Extract 3×4 projection matrix (row-major, 12 floats) from a view matrix.
    private func projectionMatrix(from viewMatrix: simd_float4x4) -> [Float] {
        var P = [Float](repeating: 0, count: 12)
        P[0]  = viewMatrix.columns.0.x; P[1]  = viewMatrix.columns.1.x
        P[2]  = viewMatrix.columns.2.x; P[3]  = viewMatrix.columns.3.x
        P[4]  = viewMatrix.columns.0.y; P[5]  = viewMatrix.columns.1.y
        P[6]  = viewMatrix.columns.2.y; P[7]  = viewMatrix.columns.3.y
        P[8]  = viewMatrix.columns.0.z; P[9]  = viewMatrix.columns.1.z
        P[10] = viewMatrix.columns.2.z; P[11] = viewMatrix.columns.3.z
        return P
    }

    private func mat3FromData(_ data: Data) -> simd_float3x3 {
        guard data.count >= 36 else { return matrix_identity_float3x3 }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return simd_float3x3(
            SIMD3<Float>(f[0], f[3], f[6]),
            SIMD3<Float>(f[1], f[4], f[7]),
            SIMD3<Float>(f[2], f[5], f[8])
        )
    }

    private func vec3FromData(_ data: Data) -> SIMD3<Float> {
        guard data.count >= 12 else { return .zero }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return SIMD3<Float>(f[0], f[1], f[2])
    }

    private func unpackFloat3(from data: Data, count: Int) -> [SIMD3<Float>] {
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return (0..<count).map { i in SIMD3<Float>(f[i*3], f[i*3+1], f[i*3+2]) }
    }

    private func averageDisplacement(pts1: [Float], pts2: [Float], count: Int) -> Float {
        guard count > 0 else { return 0 }
        var total: Float = 0
        for i in 0..<count {
            let dx = pts2[i*2] - pts1[i*2]
            let dy = pts2[i*2+1] - pts1[i*2+1]
            total += sqrt(dx*dx + dy*dy)
        }
        return total / Float(count)
    }

    private func sampleColor(_ image: UIImage, x: Float, y: Float) -> SIMD3<Float> {
        guard let cg = image.cgImage else { return SIMD3<Float>(0.5, 0.5, 0.5) }
        let iw = CGFloat(cg.width), ih = CGFloat(cg.height)
        let px = Int(CGFloat(x) / CGFloat(procW) * iw)
        let py = Int(CGFloat(y) / CGFloat(procH) * ih)
        guard px >= 0, py >= 0, px < Int(iw), py < Int(ih) else {
            return SIMD3<Float>(0.5, 0.5, 0.5)
        }
        var pixel = [UInt8](repeating: 0, count: 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixel, width: 1, height: 1,
            bitsPerComponent: 8, bytesPerRow: 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return SIMD3<Float>(0.5, 0.5, 0.5) }
        ctx.draw(cg, in: CGRect(x: -CGFloat(px), y: -CGFloat(py), width: iw, height: ih))
        return SIMD3<Float>(Float(pixel[0])/255, Float(pixel[1])/255, Float(pixel[2])/255)
    }
}

// MARK: - SIMD Helpers

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

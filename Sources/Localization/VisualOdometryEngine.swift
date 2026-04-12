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
    /// Number of keyframes currently in the active state (recent enough to auto-check).
    @Published private(set) var activeKeyframeCount: Int = 0
    /// Number of keyframes currently dormant (only woken by thumbnail similarity).
    @Published private(set) var dormantKeyframeCount: Int = 0

    private var lastMarkerFrame: Int = -999

    let pointCloud = PointCloud()
    private let visualMap = VisualMap()

    /// Descriptor dimension currently in use (64 for XFeat, 128 for ALIKED, 0 = none).
    /// If this changes, the visual map is automatically cleared and tracking restarts.
    private var activeDescDim: Int = 0

    // MARK: - Tunable Parameters

    /// ArUco marker ID to use for bootstrapping. -1 = accept any marker.
    var targetMarkerID: Int = -1
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
    /// Frames between map culling passes. 0 = disabled.
    var cullInterval: Int = 60
    /// Frames between Local Bundle Adjustment runs. 0 = disabled.
    var baInterval: Int = 30
    /// Number of recent keyframes to include in the BA window.
    var baWindowSize: Int = 7
    /// Minimum parallax (px) needed for tracking-phase triangulation.
    var minTriParallax: Float = 8.0
    /// Minimum common ALIKED tracks needed for tracking-phase triangulation.
    var minTriCommon: Int = 10
    /// Maximum reprojection error (px) to accept a triangulated point.
    var maxReprojError: Float = 2.5
    /// Minimum KF gap (in keyframe count, not frames) between current KF and loop candidate.
    var loopClosureMinKFGap: Int = 5
    /// Minimum thumbnail cosine similarity to attempt geometry verification.
    var loopSimThreshold: Float = 0.82
    /// Minimum PnP inliers required to confirm a loop (stricter than regular recovery).
    var loopMinInliers: Int = 15

    let procW: Float = 960
    let procH: Float = 720

    // MARK: - Bootstrap Complete Flag

    /// True once the first successful triangulation has completed.
    /// When true, ArUco detection is completely disabled (marker treated as
    /// regular feature points by PointTracker).
    private var bootstrapComplete: Bool = false

    // MARK: - Recovery Keyframe Database

    /// One entry in the multi-keyframe database.
    private struct StoredKeyframe {
        let image: UIImage
        /// Camera-to-world pose at the time this keyframe was saved.
        var pose: simd_float4x4
        /// L2-normalised 64×48 grayscale vector — used for fast dormant-KF similarity gating.
        let thumbnail: [Float]
        /// Visual map stable ID → 2D pixel position in the keyframe image.
        let map2D: [(mapID: Int, pos2D: SIMD2<Float>)]
        let descDim: Int
        /// Frame index at which this keyframe was saved.
        let savedAtFrame: Int
        /// Cached features (populated lazily on first recovery attempt).
        var cachedALIKED: ALIKEDFeatures?
        var cachedXFeat:  XFeatFeatures?
    }

    private var keyframes: [StoredKeyframe] = []
    /// Counts up each frame while in .lost; reset to 0 on successful recovery.
    private var lostFrameCount: Int = 0

    /// Published image of the most recently saved keyframe (for on-device debugging).
    @Published private(set) var recoveryKeyframeImage: UIImage?

    private var lastKeyframeFrame: Int = -999
    /// Minimum frames between keyframe saves during tracking.
    var keyframeInterval: Int = 30
    /// Minimum frames between periodic drift-correction runs.
    var correctionInterval: Int = 60
    /// Maximum pixel distance to associate a matched keyframe point to a map entry.
    var kfProximityPx: Float = 20.0
    /// Maximum keyframes to keep (oldest removed when exceeded).
    var maxKeyframes: Int = 20
    /// Frames after which a keyframe transitions to dormant.
    var dormancyDelay: Int = 150
    /// During lost: check dormant KFs every N frames (thumbnail-similarity gated).
    var dormantCheckInterval: Int = 5

    private static let thumbW = 64
    private static let thumbH = 48

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

    /// Guard flags to prevent concurrent map-growth / correction / BA tasks from stacking up.
    private var isGrowingMap:    Bool = false
    private var isCorrecting:    Bool = false
    private var isRunningBA:     Bool = false
    private var isDetectingLoop: Bool = false
    /// Consecutive successful PnP frames — used to gate the pre-filter threshold.
    /// Resets to 0 whenever the pose is freshly recovered (ArUco / KF recovery).
    private var stablePnPFrames: Int = 0
    /// Consecutive PnP failures — go lost only after this exceeds the hysteresis limit.
    private var consecutivePnPFailures: Int = 0
    /// Incremented on every reset(). Fire-and-forget Tasks compare against this to
    /// detect that the engine was reset while they were awaiting background work,
    /// and bail out instead of writing stale data.
    private var sessionID: Int = 0

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
        bootstrapComplete = false
        keyframes = []
        lostFrameCount = 0
        isGrowingMap           = false
        isCorrecting           = false
        isRunningBA            = false
        isDetectingLoop        = false
        stablePnPFrames        = 0
        consecutivePnPFailures = 0
        sessionID += 1
        recoveryKeyframeImage = nil
        activeKeyframeCount = 0
        dormantKeyframeCount = 0
        lastKeyframeFrame = -999
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
    ) async {
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
        case .searching:
            tryMarkerInit(procImage: procImage, aliked: topPts, intrinsics: intrinsics, descDim: descDim)
        case .lost:
            if bootstrapComplete {
                // Fast path: try ArUco re-localization first (cheap, metric-accurate).
                // The world frame IS the marker frame, so ArUco pose = direct cameraPose.
                if let arucoPose = detectArUco(procImage: procImage, intrinsics: intrinsics) {
                    cameraPose = arucoPose
                    phase = .tracking
                    lostFrameCount = 0
                    lastMarkerFrame = frameIndex
                    isMarkerInView  = true
                    stablePnPFrames        = 0   // pose is fresh — relax pre-filter until stable
                    consecutivePnPFailures = 0
                    statusMessage = "ArUco再検出 — トラッキング再開"
                    return
                }
                // Fallback: descriptor-based recovery against stored keyframes
                await tryDescriptorRecovery(procImage: procImage, intrinsics: intrinsics)
            } else {
                tryMarkerInit(procImage: procImage, aliked: topPts, intrinsics: intrinsics, descDim: descDim)
            }
        case .bootstrapping:
            await tryBootstrap(aliked: topPts, intrinsics: intrinsics, procImage: procImage, descDim: descDim)
        case .tracking:
            await trackingStep(aliked: topPts, intrinsics: intrinsics, procImage: procImage, descDim: descDim)
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
    ) async {
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
        let bootDescs = bootstrapDescriptors, bootPos = bootstrapPositions2D
        let curAliked = aliked, curDescDim = descDim
        let (pts1, pts2, common) = await Task.detached(priority: .userInitiated) {
            Self.matchBootstrapDescriptors(
                current: curAliked, descDim: curDescDim,
                bootstrapDescriptors: bootDescs, bootstrapPositions2D: bootPos
            )
        }.value

        print("[VO] Bootstrap: boot=\(bootN) cur=\(curN) matched=\(common.count) need=\(minBootstrapCommon)")

        guard common.count >= minBootstrapCommon else {
            statusMessage = "共通点不足 \(common.count)/\(minBootstrapCommon) [boot:\(bootN) cur:\(curN)]"
            return
        }

        let parallax = Self.averageDisplacement(pts1: pts1, pts2: pts2, count: common.count)
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
        let tPose1Boot = bPose, tPose2Boot = currentPose
        let tPts1Boot = pts1, tPts2Boot = pts2, tCommonBoot = common
        let tIntrBoot = intrinsics, tImgBoot = procImage, tDimBoot = descDim
        let maxErrBoot: Float = 2.0, pwBoot = procW, phBoot = procH  // tighter threshold for bootstrap quality
        let result = await Task.detached(priority: .userInitiated) {
            Self.triangulate(
                pose1: tPose1Boot, pose2: tPose2Boot,
                pts1: tPts1Boot, pts2: tPts2Boot,
                aliked: tCommonBoot,
                intrinsics: tIntrBoot,
                procImage: tImgBoot,
                confidence: 0.5,
                descDim: tDimBoot,
                maxReprojError: maxErrBoot, procW: pwBoot, procH: phBoot
            )
        }.value

        guard result.mapEntries.count >= 8 else {
            statusMessage = "三角測量点が少なすぎます (\(result.mapEntries.count)) — もっと移動してください"
            return
        }

        visualMap.add(result.mapEntries, currentFrame: frameIndex)
        pointCloud.add(result.cloud3D)
        mapPointCount = visualMap.count
        lastTriangulatedIDs = Set(result.triangulatedIDs)

        cameraPose = currentPose
        triRefPose = currentPose
        triRefPositions = Dictionary(uniqueKeysWithValues: aliked.map { ($0.id, $0.position) })
        lastTriFrame = frameIndex

        bootstrapComplete = true
        lostFrameCount = 0
        phase = .tracking

        // Save initial recovery keyframe immediately after bootstrap so that
        // recovery is possible even if the user gets lost before keyframeInterval frames.
        let initialMatches = (0..<min(common.count, result.mapEntries.count)).map {
            (entryIdx: visualMap.count - result.mapEntries.count + $0, queryIdx: $0)
        }
        addKeyframe(from: initialMatches, aliked: common, procImage: procImage, descDim: descDim)

        statusMessage = "初期化完了! マップ: \(mapPointCount)点"
    }

    // MARK: - Phase: Tracking

    private func trackingStep(
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        descDim: Int
    ) async {
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

        // Background: expensive cblas_sgemm matrix multiply
        let mapEntries = visualMap.entries
        let matches = await Task.detached(priority: .userInitiated) {
            VisualMap.computeMatches(
                queryDescriptors: queryDescs, queryCount: n,
                descDim: descDim, minScore: 0.76, entries: mapEntries
            )
        }.value
        matchCount = matches.count
        // Record which tracked-point IDs are currently matched to the map
        lastMatchedIDs = Set(matches.map { aliked[$0.queryIdx].id })
        // Update per-entry observation stats for culling
        visualMap.markObserved(indices: matches.map { $0.entryIdx }, frame: frameIndex)

        guard matches.count >= 6 else {
            print("[VO] Track: match=\(matches.count)/6 aliked=\(n) map=\(visualMap.count) → lost")
            phase = .lost
            statusMessage = "マッチ不足 (\(matches.count)/6) — ロスト"
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

        // Pre-filter by reprojection using the current pose estimate.
        // Threshold relaxes for the first few frames after a recovery to avoid rejecting
        // all observations when the recovered pose has small inaccuracies.
        if let pose = cameraPose {
            let viewMat = pose.inverse
            var fPts3D = [Float](), fPts2D = [Float]()
            fPts3D.reserveCapacity(pts3D.count)
            fPts2D.reserveCapacity(pts2D.count)
            // Fresh pose (just recovered): use 25 px — stable tracking: use 12 px
            let reprPx: Float = stablePnPFrames >= 5 ? 12 : 25
            let reprThreshSq = reprPx * reprPx
            let n = pts3D.count / 3
            for i in 0..<n {
                let pc = viewMat * SIMD4<Float>(pts3D[i*3], pts3D[i*3+1], pts3D[i*3+2], 1)
                guard pc.z > 0 else { continue }
                let u = fx * pc.x / pc.z + cx
                let v = fy * pc.y / pc.z + cy
                let dx = u - pts2D[i*2], dy = v - pts2D[i*2+1]
                guard dx*dx + dy*dy < reprThreshSq else { continue }
                fPts3D.append(pts3D[i*3]); fPts3D.append(pts3D[i*3+1]); fPts3D.append(pts3D[i*3+2])
                fPts2D.append(pts2D[i*2]); fPts2D.append(pts2D[i*2+1])
            }
            if fPts3D.count / 3 >= minPnPInliers { pts3D = fPts3D; pts2D = fPts2D }
        }

        let pnpCount = pts3D.count / 3
        let pnp = OpenCVBridge.solvePnP(
            points3D: Data(bytes: pts3D, count: pts3D.count * 4),
            points2D: Data(bytes: pts2D, count: pts2D.count * 4),
            count: pnpCount,
            fx: fx, fy: fy, cx: cx, cy: cy,
            iterations: 60  // tracking: fewer iterations for real-time performance
        )

        guard pnp.success, pnp.inlierCount >= minPnPInliers else {
            consecutivePnPFailures += 1
            stablePnPFrames = 0
            print("[VO] PnP failed: inliers=\(pnp.inlierCount)/\(minPnPInliers) success=\(pnp.success) (fail#\(consecutivePnPFailures))")
            // Require 3 consecutive failures before going lost (hysteresis)
            if consecutivePnPFailures >= 3 {
                phase = .lost
                statusMessage = "PnP失敗 (インライア: \(pnp.inlierCount)/\(minPnPInliers)) — ロスト"
            }
            return
        }

        consecutivePnPFailures = 0
        stablePnPFrames = min(stablePnPFrames + 1, 10)

        let vm = Self.buildViewMatrix(rData: pnp.rotationMatrix, tData: pnp.translationVector)
        let newPose = vm.inverse  // camera-to-world

        cameraPose = newPose
        pnpInlierCount = pnp.inlierCount

        // Update dormancy counts (cheap: just integer comparisons)
        let kfActive = keyframes.filter { frameIndex - $0.savedAtFrame <= dormancyDelay }.count
        activeKeyframeCount  = kfActive
        dormantKeyframeCount = keyframes.count - kfActive

        statusMessage = "追跡中 — PnP: \(pnp.inlierCount)点 | マップ: \(mapPointCount)点 | KF: \(activeKeyframeCount)A/\(dormantKeyframeCount)D"

        // Periodically save a recovery keyframe (ALIKED/XFeat descriptors)
        if frameIndex - lastKeyframeFrame >= keyframeInterval, !matches.isEmpty {
            addKeyframe(from: matches, aliked: aliked, procImage: procImage, descDim: descDim)
            // After KF is saved, check for loop closure against older keyframes.
            if keyframes.count > loopClosureMinKFGap, !isDetectingLoop {
                isDetectingLoop = true
                let snapIntr = intrinsics, sid = sessionID
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    await self.detectAndApplyLoopClosure(intrinsics: snapIntr, sessionID: sid)
                    self.isDetectingLoop = false
                }
            }
        }

        // Periodic drift correction: fire-and-forget so processFrame returns immediately.
        if correctionInterval > 0, frameIndex % correctionInterval == 0, !isCorrecting {
            isCorrecting = true
            let snapAliked = aliked, snapIntr = intrinsics, snapDim = descDim
            let sid = sessionID
            Task { @MainActor [weak self] in
                guard let self else { return }
                await self.periodicCorrection(aliked: snapAliked, intrinsics: snapIntr, descDim: snapDim,
                                              sessionID: sid)
                self.isCorrecting = false
            }
        }

        // Periodically triangulate new map points — fire-and-forget so processFrame returns immediately.
        if frameIndex - lastTriFrame >= triInterval, !isGrowingMap {
            isGrowingMap = true
            lastTriFrame = frameIndex  // prevent re-triggering while the task is in flight
            let snapPose = newPose, snapAliked = aliked, snapIntr = intrinsics
            let snapImg = procImage, snapDim = descDim
            let sid = sessionID
            Task { @MainActor [weak self] in
                guard let self else { return }
                await self.growMap(currentPose: snapPose, aliked: snapAliked,
                                   intrinsics: snapIntr, procImage: snapImg, descDim: snapDim,
                                   sessionID: sid)
                self.isGrowingMap = false
            }
        }

        // Periodically run Local Bundle Adjustment.
        if baInterval > 0, frameIndex % baInterval == 0, !isRunningBA,
           keyframes.count >= 2 {
            isRunningBA = true
            let sid = sessionID
            Task { @MainActor [weak self] in
                guard let self else { return }
                await self.localBAStep(intrinsics: intrinsics, sessionID: sid)
                self.isRunningBA = false
            }
        }

        // Periodically cull stale map points (skip the frame BA just fired to avoid index race).
        if cullInterval > 0, frameIndex % cullInterval == 0,
           !(baInterval > 0 && frameIndex % baInterval == 0) {
            let removed = visualMap.cull(currentFrame: frameIndex)
            if removed > 0 {
                mapPointCount = visualMap.count
                print("[VO] Culled \(removed) stale map points — remaining: \(mapPointCount)")
            }
        }
    }

    // MARK: - Map Growth

    private func growMap(
        currentPose: simd_float4x4,
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        descDim: Int,
        sessionID sid: Int = -1
    ) async {
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

        let parallax = Self.averageDisplacement(pts1: pts1, pts2: pts2, count: common.count)
        guard parallax >= minTriParallax else { return }

        let tPose1 = refPose, tPose2 = currentPose
        let tPts1 = pts1, tPts2 = pts2, tCommon = common
        let tIntr = intrinsics, tImg = procImage, tDim = descDim
        let maxErr = maxReprojError, pw = procW, ph = procH
        let result = await Task.detached(priority: .userInitiated) {
            Self.triangulate(
                pose1: tPose1, pose2: tPose2, pts1: tPts1, pts2: tPts2,
                aliked: tCommon, intrinsics: tIntr, procImage: tImg,
                confidence: 0.6, descDim: tDim,
                maxReprojError: maxErr, procW: pw, procH: ph
            )
        }.value

        // Bail out if reset() was called or we're no longer tracking (phase changed during await)
        guard sid == -1 || sessionID == sid else { return }
        guard phase == .tracking else { return }

        print("[VO] GrowMap: common=\(common.count) parallax=\(String(format:"%.1f",parallax))px tri=\(result.mapEntries.count)")
        guard !result.mapEntries.isEmpty else { return }

        visualMap.add(result.mapEntries, currentFrame: frameIndex)
        pointCloud.add(result.cloud3D)
        mapPointCount = visualMap.count
        lastTriangulatedIDs = Set(result.triangulatedIDs)

        // Advance reference keyframe so baseline grows for next attempt
        triRefPose = currentPose
        triRefPositions = Dictionary(uniqueKeysWithValues: aliked.map { ($0.id, $0.position) })
        lastTriFrame = frameIndex
    }

    // MARK: - Local Bundle Adjustment

    /// Runs a sparse Local Bundle Adjustment over the most recent BA window of keyframes.
    ///
    /// - Collects unique 3-D landmarks observed across the window keyframes.
    /// - Builds 2-D observations for each (keyframe, landmark) pair.
    /// - Calls the Ceres-based `runLocalBA` C++ function.
    /// - Writes refined landmark positions back into `visualMap`.
    /// - Updates stored keyframe poses and the live `cameraPose`.
    private func localBAStep(
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        sessionID sid: Int
    ) async {
        guard keyframes.count >= 2 else { return }

        // Window: last baWindowSize keyframes (or all if fewer).
        let window = Array(keyframes.suffix(baWindowSize))
        guard window.count >= 2 else { return }

        // --- Collect unique map IDs referenced in the window ---
        var mapIDSet = Set<Int>()
        for kf in window {
            for obs in kf.map2D { mapIDSet.insert(obs.mapID) }
        }

        // Look up current 3-D positions for each map ID
        let idToPosCurrent = visualMap.idToPositionSnapshot()
        let validIDs = mapIDSet.filter { idToPosCurrent[$0] != nil }
        guard validIDs.count >= 4 else { return }

        // Build stable index mapping: mapID → BA point index
        let sortedIDs = Array(validIDs)
        var mapIDToBAIdx = [Int: Int]()
        for (i, id) in sortedIDs.enumerated() { mapIDToBAIdx[id] = i }

        // --- Pack poses (camera-to-world) ---
        var posesC = window.map { $0.pose }
        let nPoses  = posesC.count
        let nPoints = sortedIDs.count

        // --- Pack 3-D points ---
        var pointsC = sortedIDs.map { id -> simd_float3 in
            idToPosCurrent[id] ?? .zero
        }

        // --- Build observations ---
        var obsArray = [BAObservation]()
        for (poseIdx, kf) in window.enumerated() {
            for obs in kf.map2D {
                guard let ptIdx = mapIDToBAIdx[obs.mapID] else { continue }
                obsArray.append(BAObservation(
                    poseIdx:  Int32(poseIdx),
                    pointIdx: Int32(ptIdx),
                    u: obs.pos2D.x,
                    v: obs.pos2D.y
                ))
            }
        }
        guard obsArray.count >= 8 else { return }

        // Run Ceres BA on a background thread
        let (fx, fy, cx, cy) = intrinsics
        let baResult: BAResult = await Task.detached(priority: .userInitiated) {
            posesC.withUnsafeBufferPointer { posPtr in
                pointsC.withUnsafeBufferPointer { ptPtr in
                    obsArray.withUnsafeBufferPointer { obsPtr in
                        runLocalBA(
                            posPtr.baseAddress, Int32(nPoses),
                            ptPtr.baseAddress,  Int32(nPoints),
                            obsPtr.baseAddress, Int32(obsArray.count),
                            fx, fy, cx, cy,
                            true  // fix first pose (gauge freedom)
                        )
                    }
                }
            }
        }.value

        // Bail out if reset() was called while we were computing
        guard sid == sessionID else {
            free(baResult.poses); free(baResult.points)
            return
        }

        guard baResult.converged else {
            print("[BA] Did not converge (cost=\(String(format:"%.4f", baResult.finalCost)))")
            free(baResult.poses); free(baResult.points)
            return
        }

        // --- Write refined 3-D positions back into the map ---
        for i in 0..<Int(baResult.pointCount) {
            let mapID = sortedIDs[i]
            visualMap.updatePosition(id: mapID, newPosition: baResult.points[i])
        }

        // --- Update keyframe poses in the window (last baWindowSize entries) ---
        let windowStart = keyframes.count - window.count
        for i in 0..<Int(baResult.poseCount) {
            let kfIdx = windowStart + i
            guard kfIdx < keyframes.count else { break }
            keyframes[kfIdx].pose = baResult.poses[i]
        }

        // --- Update live camera pose from the last refined window pose ---
        if baResult.poseCount > 0 {
            cameraPose = baResult.poses[Int(baResult.poseCount) - 1]
        }

        free(baResult.poses)
        free(baResult.points)

        print("[BA] Converged: cost=\(String(format:"%.4f", baResult.finalCost)) pts=\(nPoints) obs=\(obsArray.count)")
    }

    // MARK: - Descriptor Recovery (multi-keyframe, ALIKED / XFeat)

    /// Try to recover from lost state by matching against stored keyframes.
    ///
    /// - Active keyframes (age ≤ dormancyDelay frames) are always tried.
    /// - Dormant keyframes are gated by thumbnail similarity to the current frame,
    ///   and only the top-2 similar ones are tried every `dormantCheckInterval` frames.
    private func tryDescriptorRecovery(
        procImage: UIImage,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) async {
        guard !keyframes.isEmpty else {
            statusMessage = "キーフレームなし — ロスト (tracking後に自動保存)"
            return
        }

        lostFrameCount += 1

        // Compute thumbnail similarity for dormant gating
        let currentThumb = makeThumbnail(from: procImage)

        var activeIndices:  [Int] = []
        var dormantSims: [(idx: Int, sim: Float)] = []
        for (i, kf) in keyframes.enumerated() {
            if frameIndex - kf.savedAtFrame <= dormancyDelay {
                activeIndices.append(i)
            } else {
                dormantSims.append((i, thumbnailSimilarity(currentThumb, kf.thumbnail)))
            }
        }
        dormantSims.sort { $0.sim > $1.sim }

        activeKeyframeCount  = activeIndices.count
        dormantKeyframeCount = dormantSims.count

        // Build candidate list
        var candidates = activeIndices
        if lostFrameCount % dormantCheckInterval == 0 {
            candidates += dormantSims.prefix(4).map { $0.idx }  // check top-4 dormant (was top-2)
        }
        // When no active KFs exist, always check top-2 dormant every frame
        if activeIndices.isEmpty, lostFrameCount % dormantCheckInterval != 0 {
            candidates += dormantSims.prefix(2).map { $0.idx }
        }
        print("[VO] Recovery: active=\(activeIndices.count) dormant=\(dormantSims.count) candidates=\(candidates.count) lostFrame=\(lostFrameCount)")

        guard !candidates.isEmpty else {
            statusMessage = "全KF休眠 — ロスト (復帰間隔待機中)"
            return
        }

        // Extract current frame features ONCE (reused across all KF attempts)
        let descDim = keyframes[candidates[0]].descDim
        let capturedImage = procImage
        let featureResult: (kps: [SIMD2<Float>], descs: [Float])? = await Task.detached(priority: .userInitiated) {
            if descDim == 128 {
                guard let f = ALIKEDMatcher.shared.extractFeatures(from: capturedImage), f.count > 0 else { return nil }
                return (f.keypoints, f.descriptors)
            } else {
                guard let f = XFeatMatcher.shared.extractFeatures(from: capturedImage), f.count > 0 else { return nil }
                return (f.keypoints, f.descriptors)
            }
        }.value
        guard let (curKPs, curDescs) = featureResult else {
            statusMessage = "現フレーム特徴量抽出失敗 — ロスト"; return
        }

        // Try each candidate keyframe
        for idx in candidates {
            if let pose = await tryRecoverFromKeyframe(
                idx: idx, curKPs: curKPs, curDescs: curDescs, intrinsics: intrinsics
            ) {
                cameraPose = pose
                phase = .tracking
                lostFrameCount = 0
                stablePnPFrames        = 0   // pose is fresh — relax pre-filter until stable
                consecutivePnPFailures = 0
                statusMessage = "KF[\(idx)]で復帰! アクティブ:\(activeKeyframeCount) 休眠:\(dormantKeyframeCount)"
                return
            }
        }

        statusMessage = "復帰失敗 — KF: \(activeKeyframeCount)アクティブ / \(dormantKeyframeCount)休眠"
    }

    /// Try to recover pose from a single stored keyframe. Returns the recovered pose or nil.
    private func tryRecoverFromKeyframe(
        idx: Int,
        curKPs:   [SIMD2<Float>],
        curDescs: [Float],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) async -> simd_float4x4? {
        guard idx < keyframes.count else { return nil }  // reset() may have cleared keyframes during prior await
        var kf = keyframes[idx]
        let descDim = kf.descDim

        // Get (or lazily extract and cache) KF features
        let kfKPs:   [SIMD2<Float>]
        let kfDescs: [Float]
        if descDim == 128 {
            if let cached = kf.cachedALIKED {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let kfImage = kf.image
                let f = await Task.detached(priority: .userInitiated) {
                    ALIKEDMatcher.shared.extractFeatures(from: kfImage)
                }.value
                guard let f, f.count > 0 else { return nil }
                guard idx < keyframes.count else { return nil }  // keyframes may have changed during await
                kf.cachedALIKED = f
                keyframes[idx] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        } else {
            if let cached = kf.cachedXFeat {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let kfImage = kf.image
                let f = await Task.detached(priority: .userInitiated) {
                    XFeatMatcher.shared.extractFeatures(from: kfImage)
                }.value
                guard let f, f.count > 0 else { return nil }
                guard idx < keyframes.count else { return nil }  // keyframes may have changed during await
                kf.cachedXFeat = f
                keyframes[idx] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        }

        let kfKPsCopy = kfKPs, kfDescsCopy = kfDescs
        let curKPsCopy = curKPs, curDescsCopy = curDescs
        let map2DCopy = kf.map2D, idToPosCopy = visualMap.idToPositionSnapshot()
        let dim = descDim, threshold: Float = 0.70
        let minInliers = minPnPInliers, proximity = kfProximityPx
        let (fx, fy, cx, cy) = intrinsics

        let poseResult: simd_float4x4? = await Task.detached(priority: .userInitiated) {
            let pairs = Self.mutualNNRaw(
                desc1: kfDescsCopy, count1: kfKPsCopy.count,
                desc2: curDescsCopy, count2: curKPsCopy.count,
                descDim: dim, threshold: threshold
            )
            guard pairs.count >= minInliers else { return nil }

            var pts3D = [Float](), pts2D = [Float]()
            for (kfPtIdx, curPtIdx) in pairs {
                let kfKP = kfKPsCopy[kfPtIdx]
                guard let best = map2DCopy.min(by: {
                    simd_length($0.pos2D - kfKP) < simd_length($1.pos2D - kfKP)
                }), simd_length(best.pos2D - kfKP) < proximity,
                let pos3D = idToPosCopy[best.mapID] else { continue }
                let curKP = curKPsCopy[curPtIdx]
                pts3D.append(pos3D.x); pts3D.append(pos3D.y); pts3D.append(pos3D.z)
                pts2D.append(curKP.x); pts2D.append(curKP.y)
            }
            let count = pts3D.count / 3
            guard count >= minInliers else { return nil }

            let pnp = OpenCVBridge.solvePnP(
                points3D: Data(bytes: pts3D, count: pts3D.count * 4),
                points2D: Data(bytes: pts2D, count: pts2D.count * 4),
                count: count, fx: fx, fy: fy, cx: cx, cy: cy,
                iterations: 150  // recovery: more iterations for reliability
            )
            guard pnp.success, pnp.inlierCount >= minInliers else { return nil }
            return Self.buildViewMatrix(rData: pnp.rotationMatrix, tData: pnp.translationVector).inverse
        }.value

        return poseResult
    }

    // MARK: - Periodic Drift Correction

    /// Re-anchor the current pose using the most recent active keyframe.
    /// KF features are cached after the first extraction, making repeated calls cheap.
    private func periodicCorrection(
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        descDim: Int,
        sessionID sid: Int = -1
    ) async {
        // Use most recent keyframe with matching descDim (active or dormant — doesn't matter)
        guard let kfIdx = keyframes.indices.last(where: { keyframes[$0].descDim == descDim }) else { return }
        let M = aliked.count
        guard M >= minPnPInliers else { return }

        var kf = keyframes[kfIdx]

        // Lazily extract and cache KF features
        let kfKPs: [SIMD2<Float>]
        let kfDescs: [Float]
        if descDim == 128 {
            if let cached = kf.cachedALIKED {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let kfImage = kf.image
                let f = await Task.detached(priority: .userInitiated) {
                    ALIKEDMatcher.shared.extractFeatures(from: kfImage)
                }.value
                guard let f, f.count > 0 else { return }
                guard kfIdx < keyframes.count else { return }  // keyframes may have changed during await
                kf.cachedALIKED = f; keyframes[kfIdx] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        } else {
            if let cached = kf.cachedXFeat {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let kfImage = kf.image
                let f = await Task.detached(priority: .userInitiated) {
                    XFeatMatcher.shared.extractFeatures(from: kfImage)
                }.value
                guard let f, f.count > 0 else { return }
                guard kfIdx < keyframes.count else { return }  // keyframes may have changed during await
                kf.cachedXFeat = f; keyframes[kfIdx] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        }

        // Build L2-normalised query matrix from tracked points
        var queryDescs = [Float](repeating: 0, count: M * descDim)
        for (i, pt) in aliked.enumerated() {
            guard pt.descriptor.count == descDim else { continue }
            var norm2: Float = 0
            vDSP_svesq(pt.descriptor, 1, &norm2, vDSP_Length(descDim))
            let scale = 1.0 / max(sqrtf(norm2), 1e-8)
            vDSP_vsmul(pt.descriptor, 1, [scale], &queryDescs[i * descDim], 1, vDSP_Length(descDim))
        }

        let kfKPsCopy = kfKPs, kfDescsCopy = kfDescs
        let map2DCopy = kf.map2D, idToPosCopy = visualMap.idToPositionSnapshot()
        let dim = descDim, queryDescsCopy = queryDescs, mCount = M
        let threshold: Float = 0.72
        let minInliers = minPnPInliers, proximity = kfProximityPx
        let (fx, fy, cx, cy) = intrinsics
        let alikedCopy = aliked

        let poseResult: simd_float4x4? = await Task.detached(priority: .userInitiated) {
            let pairs = Self.mutualNNRaw(
                desc1: kfDescsCopy, count1: kfKPsCopy.count,
                desc2: queryDescsCopy, count2: mCount,
                descDim: dim, threshold: threshold
            )
            guard pairs.count >= minInliers else { return nil }

            var pts3D = [Float]()
            var pts2D = [Float]()
            for (kfPtIdx, queryIdx) in pairs {
                let kfKP = kfKPsCopy[kfPtIdx]
                guard let best = map2DCopy.min(by: {
                    simd_length($0.pos2D - kfKP) < simd_length($1.pos2D - kfKP)
                }), simd_length(best.pos2D - kfKP) < proximity,
                let pos3D = idToPosCopy[best.mapID] else { continue }
                let pt = alikedCopy[queryIdx]
                pts3D.append(pos3D.x); pts3D.append(pos3D.y); pts3D.append(pos3D.z)
                pts2D.append(pt.position.x); pts2D.append(pt.position.y)
            }
            let count = pts3D.count / 3
            guard count >= minInliers else { return nil }

            let pnp = OpenCVBridge.solvePnP(
                points3D: Data(bytes: pts3D, count: pts3D.count * 4),
                points2D: Data(bytes: pts2D, count: pts2D.count * 4),
                count: count, fx: fx, fy: fy, cx: cx, cy: cy,
                iterations: 100  // correction: moderate iterations
            )
            guard pnp.success, pnp.inlierCount >= minInliers else { return nil }
            return Self.buildViewMatrix(rData: pnp.rotationMatrix, tData: pnp.translationVector).inverse
        }.value

        // Bail out if reset() was called while we were awaiting background work
        guard sid == -1 || sessionID == sid else { return }

        if let pose = poseResult {
            cameraPose = pose
            print("[VO] Correction KF[\(kfIdx)]: success")
        }
    }

    // MARK: - Keyframe Helpers

    /// Add a new keyframe to the database from the current PnP matches.
    private func addKeyframe(
        from matches: [(entryIdx: Int, queryIdx: Int)],
        aliked: [TrackedPoint],
        procImage: UIImage,
        descDim: Int
    ) {
        let map2D = matches.compactMap { m -> (mapID: Int, pos2D: SIMD2<Float>)? in
            guard m.queryIdx < aliked.count, m.entryIdx < visualMap.count else { return nil }
            return (mapID: visualMap.id(at: m.entryIdx), pos2D: aliked[m.queryIdx].position)
        }
        guard !map2D.isEmpty, let pose = cameraPose else { return }

        let kf = StoredKeyframe(
            image: procImage,
            pose: pose,
            thumbnail: makeThumbnail(from: procImage),
            map2D: map2D,
            descDim: descDim,
            savedAtFrame: frameIndex,
            cachedALIKED: nil,
            cachedXFeat: nil
        )
        keyframes.append(kf)
        if keyframes.count > maxKeyframes { keyframes.removeFirst() }

        recoveryKeyframeImage = procImage
        lastKeyframeFrame = frameIndex

        let active = keyframes.filter { frameIndex - $0.savedAtFrame <= dormancyDelay }.count
        activeKeyframeCount  = active
        dormantKeyframeCount = keyframes.count - active
    }

    /// Build a small L2-normalised grayscale thumbnail from an image.
    private func makeThumbnail(from image: UIImage) -> [Float] {
        let W = Self.thumbW, H = Self.thumbH
        let grayData = OpenCVBridge.toGray(image, width: Int32(W), height: Int32(H))
        let n = W * H
        var v = [Float](repeating: 0, count: n)
        grayData.withUnsafeBytes { ptr in
            let bytes = ptr.bindMemory(to: UInt8.self)
            for i in 0..<n { v[i] = Float(bytes[i]) }
        }
        var norm2: Float = 0
        vDSP_svesq(v, 1, &norm2, vDSP_Length(n))
        let scale = 1.0 / max(sqrtf(norm2), 1e-8)
        vDSP_vsmul(v, 1, [scale], &v, 1, vDSP_Length(n))
        return v
    }

    /// Cosine similarity between two L2-normalised thumbnail vectors.
    private func thumbnailSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(min(a.count, b.count)))
        return dot
    }

    /// Mutual nearest-neighbour matching between two descriptor matrices using BLAS.
    /// Returns pairs (idx in desc1, idx in desc2) that are mutual nearest neighbours
    /// with cosine similarity ≥ threshold. Descriptors must be L2-normalised.
    nonisolated private static func mutualNNRaw(
        desc1: [Float], count1: Int,
        desc2: [Float], count2: Int,
        descDim: Int, threshold: Float
    ) -> [(Int, Int)] {
        let N = count1, M = count2, D = descDim
        guard N > 0, M > 0, D > 0 else { return [] }

        var S = [Float](repeating: 0, count: N * M)
        desc1.withUnsafeBufferPointer { p1 in
            desc2.withUnsafeBufferPointer { p2 in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            Int32(N), Int32(M), Int32(D),
                            1.0, p1.baseAddress!, Int32(D),
                                 p2.baseAddress!, Int32(D),
                            0.0, &S, Int32(M))
            }
        }

        var nn1 = [Int](repeating: -1, count: N)
        for i in 0..<N {
            var best: Float = threshold - 0.001; var bestJ = -1
            for j in 0..<M { if S[i*M+j] > best { best = S[i*M+j]; bestJ = j } }
            nn1[i] = bestJ
        }
        var nn2 = [Int](repeating: -1, count: M)
        for j in 0..<M {
            var best: Float = -1; var bestI = -1
            for i in 0..<N { if S[i*M+j] > best { best = S[i*M+j]; bestI = i } }
            nn2[j] = bestI
        }

        var result: [(Int, Int)] = []
        for i in 0..<N {
            let j = nn1[i]
            guard j >= 0, nn2[j] == i else { continue }
            result.append((i, j))
        }
        return result
    }

    // MARK: - Triangulation

    private struct TriResult {
        let mapEntries: [VisualMapEntry]
        let cloud3D: [Point3D]
        let triangulatedIDs: [Int]  // TrackedPoint IDs of successfully triangulated points
    }

    nonisolated private static func triangulate(
        pose1: simd_float4x4,
        pose2: simd_float4x4,
        pts1: [Float],
        pts2: [Float],
        aliked: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        procImage: UIImage,
        confidence: Float,
        descDim: Int,
        maxReprojError: Float,
        procW: Float,
        procH: Float
    ) -> TriResult {
        let (fx, fy, cx, cy) = intrinsics
        let count = aliked.count
        guard count > 0, pts1.count == count * 2, pts2.count == count * 2 else {
            return TriResult(mapEntries: [], cloud3D: [], triangulatedIDs: [])
        }

        let V1 = pose1.inverse  // world-to-camera
        let V2 = pose2.inverse

        let P1 = Self.projectionMatrix(from: V1)
        let P2 = Self.projectionMatrix(from: V2)

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

        let worldPts = Self.unpackFloat3(from: raw, count: count)

        var mapEntries: [VisualMapEntry] = []
        var cloud3D: [Point3D] = []
        var triangulatedIDs: [Int] = []

        for (i, wp) in worldPts.enumerated() {
            guard wp != .zero else { continue }

            // Cheirality + depth range check in both cameras
            let c1 = V1 * SIMD4<Float>(wp, 1)
            let c2 = V2 * SIMD4<Float>(wp, 1)
            guard c1.z > 0.05, c1.z < 50, c2.z > 0.05, c2.z < 50 else { continue }

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

            let color = Self.sampleColor(procImage, x: pts2[i*2], y: pts2[i*2+1], procW: procW, procH: procH)
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
    nonisolated private static func matchBootstrapDescriptors(
        current: [TrackedPoint], descDim: Int,
        bootstrapDescriptors: [Float], bootstrapPositions2D: [SIMD2<Float>]
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
    ///
    /// Available in all phases — used for both initial bootstrap and re-localization
    /// when lost. If `targetMarkerID >= 0`, only accepts the marker with that specific ID.
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

        // Filter by target marker ID if specified
        if targetMarkerID >= 0, aruco.markerId != targetMarkerID { return nil }

        let vm = Self.buildViewMatrix(rData: aruco.rotationMatrix, tData: aruco.translationVector)
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
        let relR = Self.mat3FromData(rp.rotationMatrix)
        let relT = Self.vec3FromData(rp.translationVector)

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
    nonisolated private static func buildViewMatrix(rData: Data, tData: Data) -> simd_float4x4 {
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
    nonisolated private static func projectionMatrix(from viewMatrix: simd_float4x4) -> [Float] {
        var P = [Float](repeating: 0, count: 12)
        P[0]  = viewMatrix.columns.0.x; P[1]  = viewMatrix.columns.1.x
        P[2]  = viewMatrix.columns.2.x; P[3]  = viewMatrix.columns.3.x
        P[4]  = viewMatrix.columns.0.y; P[5]  = viewMatrix.columns.1.y
        P[6]  = viewMatrix.columns.2.y; P[7]  = viewMatrix.columns.3.y
        P[8]  = viewMatrix.columns.0.z; P[9]  = viewMatrix.columns.1.z
        P[10] = viewMatrix.columns.2.z; P[11] = viewMatrix.columns.3.z
        return P
    }

    nonisolated private static func mat3FromData(_ data: Data) -> simd_float3x3 {
        guard data.count >= 36 else { return matrix_identity_float3x3 }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return simd_float3x3(
            SIMD3<Float>(f[0], f[3], f[6]),
            SIMD3<Float>(f[1], f[4], f[7]),
            SIMD3<Float>(f[2], f[5], f[8])
        )
    }

    nonisolated private static func vec3FromData(_ data: Data) -> SIMD3<Float> {
        guard data.count >= 12 else { return .zero }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return SIMD3<Float>(f[0], f[1], f[2])
    }

    nonisolated private static func unpackFloat3(from data: Data, count: Int) -> [SIMD3<Float>] {
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return (0..<count).map { i in SIMD3<Float>(f[i*3], f[i*3+1], f[i*3+2]) }
    }

    nonisolated private static func averageDisplacement(pts1: [Float], pts2: [Float], count: Int) -> Float {
        guard count > 0 else { return 0 }
        var total: Float = 0
        for i in 0..<count {
            let dx = pts2[i*2] - pts1[i*2]
            let dy = pts2[i*2+1] - pts1[i*2+1]
            total += sqrt(dx*dx + dy*dy)
        }
        return total / Float(count)
    }

    nonisolated private static func sampleColor(_ image: UIImage, x: Float, y: Float, procW: Float, procH: Float) -> SIMD3<Float> {
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

    // MARK: - Loop Closure

    /// Scan older keyframes for a loop using thumbnail similarity, then verify geometrically.
    /// On success, apply linear position drift correction over the affected KF range.
    private func detectAndApplyLoopClosure(
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        sessionID sid: Int
    ) async {
        let n = keyframes.count - 1
        guard n > loopClosureMinKFGap else { return }

        let currentKF = keyframes[n]
        let searchEnd = n - loopClosureMinKFGap

        // Step 1: Thumbnail scan — find the most similar older KF
        var bestSim: Float = loopSimThreshold
        var bestIdx = -1
        for j in 0...searchEnd {
            let s = thumbnailSimilarity(currentKF.thumbnail, keyframes[j].thumbnail)
            if s > bestSim { bestSim = s; bestIdx = j }
        }
        guard bestIdx >= 0 else { return }

        print("[LC] Candidate KF[\(n)]↔KF[\(bestIdx)] sim=\(String(format:"%.3f", bestSim))")

        // Step 2: Extract features from the current KF image
        let descDim = currentKF.descDim
        let kfImage = currentKF.image
        let curFeatures: (kps: [SIMD2<Float>], descs: [Float])? = await Task.detached(priority: .utility) {
            if descDim == 128 {
                guard let f = ALIKEDMatcher.shared.extractFeatures(from: kfImage), f.count > 0 else { return nil }
                return (f.keypoints, f.descriptors)
            } else {
                guard let f = XFeatMatcher.shared.extractFeatures(from: kfImage), f.count > 0 else { return nil }
                return (f.keypoints, f.descriptors)
            }
        }.value
        guard let (curKPs, curDescs) = curFeatures else { return }
        guard sid == sessionID else { return }

        // Step 3: Geometry verification — match loopKF features against currentKF features
        let loopPose = await verifyLoop(
            loopKFIdx: bestIdx,
            curKPs: curKPs, curDescs: curDescs,
            intrinsics: intrinsics
        )
        guard let loopPose else {
            print("[LC] Geometry verification failed KF[\(n)]↔KF[\(bestIdx)]")
            return
        }
        guard sid == sessionID else { return }

        print("[LC] Loop confirmed! Applying drift correction KF[\(bestIdx)]→KF[\(n)]")

        // Step 4: Apply linear position drift correction
        applyLoopDrift(currentKFIdx: n, loopKFIdx: bestIdx, correctedPose: loopPose)
    }

    /// Verify loop geometry: match loop KF's stored features against curKPs/curDescs, solve PnP.
    private func verifyLoop(
        loopKFIdx j: Int,
        curKPs: [SIMD2<Float>], curDescs: [Float],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float)
    ) async -> simd_float4x4? {
        guard j < keyframes.count else { return nil }
        var kf = keyframes[j]
        let descDim = kf.descDim

        // Get (or lazily extract and cache) loop KF features
        let kfKPs: [SIMD2<Float>]
        let kfDescs: [Float]
        if descDim == 128 {
            if let cached = kf.cachedALIKED {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let img = kf.image
                let f = await Task.detached(priority: .utility) {
                    ALIKEDMatcher.shared.extractFeatures(from: img)
                }.value
                guard let f, f.count > 0 else { return nil }
                guard j < keyframes.count else { return nil }
                kf.cachedALIKED = f; keyframes[j] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        } else {
            if let cached = kf.cachedXFeat {
                kfKPs = cached.keypoints; kfDescs = cached.descriptors
            } else {
                let img = kf.image
                let f = await Task.detached(priority: .utility) {
                    XFeatMatcher.shared.extractFeatures(from: img)
                }.value
                guard let f, f.count > 0 else { return nil }
                guard j < keyframes.count else { return nil }
                kf.cachedXFeat = f; keyframes[j] = kf
                kfKPs = f.keypoints; kfDescs = f.descriptors
            }
        }

        let kfKPsCopy = kfKPs, kfDescsCopy = kfDescs
        let curKPsCopy = curKPs, curDescsCopy = curDescs
        let map2DCopy = kf.map2D, idToPosCopy = visualMap.idToPositionSnapshot()
        let dim = descDim, proximity = kfProximityPx, minInliers = loopMinInliers
        let (fx, fy, cx, cy) = intrinsics

        return await Task.detached(priority: .utility) {
            let pairs = Self.mutualNNRaw(
                desc1: kfDescsCopy, count1: kfKPsCopy.count,
                desc2: curDescsCopy, count2: curKPsCopy.count,
                descDim: dim, threshold: 0.70
            )
            guard pairs.count >= minInliers else { return nil }

            var pts3D = [Float](), pts2D = [Float]()
            for (kfPtIdx, curPtIdx) in pairs {
                let kfKP = kfKPsCopy[kfPtIdx]
                guard let best = map2DCopy.min(by: {
                    simd_length($0.pos2D - kfKP) < simd_length($1.pos2D - kfKP)
                }), simd_length(best.pos2D - kfKP) < proximity,
                let pos3D = idToPosCopy[best.mapID] else { continue }
                let curKP = curKPsCopy[curPtIdx]
                pts3D.append(pos3D.x); pts3D.append(pos3D.y); pts3D.append(pos3D.z)
                pts2D.append(curKP.x); pts2D.append(curKP.y)
            }
            let count = pts3D.count / 3
            guard count >= minInliers else { return nil }

            let pnp = OpenCVBridge.solvePnP(
                points3D: Data(bytes: pts3D, count: pts3D.count * 4),
                points2D: Data(bytes: pts2D, count: pts2D.count * 4),
                count: count, fx: fx, fy: fy, cx: cx, cy: cy,
                iterations: 150  // loop closure: high iterations for reliability
            )
            guard pnp.success, pnp.inlierCount >= minInliers else { return nil }
            return Self.buildViewMatrix(rData: pnp.rotationMatrix, tData: pnp.translationVector).inverse
        }.value
    }

    /// Apply linear position drift correction to keyframes in range (loopKFIdx, currentKFIdx].
    /// The loop KF itself is kept fixed; each subsequent KF receives a linearly weighted
    /// share of the total positional drift.
    private func applyLoopDrift(
        currentKFIdx n: Int,
        loopKFIdx j: Int,
        correctedPose: simd_float4x4
    ) {
        guard n > j, n < keyframes.count else { return }
        let driftT = SIMD3<Float>(
            correctedPose.columns.3.x - keyframes[n].pose.columns.3.x,
            correctedPose.columns.3.y - keyframes[n].pose.columns.3.y,
            correctedPose.columns.3.z - keyframes[n].pose.columns.3.z
        )
        let total = Float(n - j)
        for k in (j + 1)...n {
            let w = Float(k - j) / total
            keyframes[k].pose.columns.3.x += driftT.x * w
            keyframes[k].pose.columns.3.y += driftT.y * w
            keyframes[k].pose.columns.3.z += driftT.z * w
        }
        cameraPose = keyframes[n].pose
        print("[LC] Drift Δt=(\(String(format:"%.3f",driftT.x)),\(String(format:"%.3f",driftT.y)),\(String(format:"%.3f",driftT.z))) KF[\(j)]→KF[\(n)]")
    }
}

// MARK: - SIMD Helpers

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

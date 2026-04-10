import Foundation
import ARKit
import UIKit
import simd

/// Visual reconstruction engine backed by ARKit VIO.
///
/// **Normal mode (ARKit tracking good)**
/// Uses `ARCamera.viewMatrix` (world-to-camera, metric) as the absolute pose
/// for each keyframe.  Two absolute projection matrices are passed to
/// `triangulatePoints`, so the output 3D points are already in ARKit world space
/// (metres, gravity-aligned).
///
/// When `markerToWorldTransform` is set, all 3D points are converted to
/// marker-relative space before being added to `pointCloud` and `visualMap`.
///
/// **Fallback mode (ARKit limited / not available)**
/// Falls back to Essential Matrix pose between consecutive keyframes, anchored to
/// the last known good ARKit view matrix.
@MainActor
final class ReconstructionEngine: ObservableObject {

    @Published var pointCloud      = PointCloud()
    @Published var frameCount      = 0
    @Published var reconstructedCount = 0
    @Published var isRecording     = false
    @Published var statusMessage   = "Press Record to start"

    // ── Tracking mode ─────────────────────────────────────────────────────
    enum TrackingSource { case arkit, fallback }
    @Published private(set) var trackingSource: TrackingSource = .arkit

    // ── Visual map (for PnP localization after ARKit is disabled) ─────────
    let visualMap = VisualMap()

    // ── Parameters ────────────────────────────────────────────────────────
    var keyframeInterval: Int = 10
    var minParallax:     Float = 5.0
    var minAlikedPoints: Int  = 30
    var markerSizeMeters: Float = 0.15

    // ── Marker-relative coordinate transform ──────────────────────────────
    /// Set by ContentView once the ArUco marker is confirmed.
    /// When set, all 3D points are stored in marker-relative space.
    var markerToWorldTransform: simd_float4x4? = nil

    /// ARKit always provides metric scale.
    var metersPerUnit: Float { 1.0 }
    var arucoScaleCalibrated: Bool { true }

    // ── Processing space ──────────────────────────────────────────────────
    let procW: Float = 960
    let procH: Float = 720

    // ── Previous keyframe state ───────────────────────────────────────────
    private var prevKeyframePositions:   [Int: SIMD2<Float>] = [:]
    private var prevKeyframeDescriptors: [Int: [Float]]      = [:]
    private var prevKeyframeViewMatrix:  simd_float4x4?
    private var prevKeyframeIntrinsics:  (fx: Float, fy: Float, cx: Float, cy: Float)?
    private var prevKeyframeArucoR:      simd_float3x3?
    private var prevKeyframeArucoT:      SIMD3<Float>?

    /// Last frame that had good ARKit tracking — anchor for fallback VO.
    private var lastGoodARKitViewMatrix: simd_float4x4?

    // MARK: - Public API

    func startRecording() {
        isRecording = true
        frameCount  = 0
        statusMessage = "Scanning…"
        prevKeyframePositions   = [:]
        prevKeyframeDescriptors = [:]
        prevKeyframeViewMatrix  = nil
        prevKeyframeIntrinsics  = nil
        prevKeyframeArucoR      = nil
        prevKeyframeArucoT      = nil
    }

    func stopRecording() {
        isRecording = false
        statusMessage = "Stopped — \(reconstructedCount) pts"
    }

    func clearCloud() {
        pointCloud.clear()
        visualMap.clear()
        reconstructedCount = 0
        prevKeyframePositions   = [:]
        prevKeyframeDescriptors = [:]
        prevKeyframeViewMatrix  = nil
        lastGoodARKitViewMatrix = nil
        statusMessage = "Cloud cleared"
    }

    /// Main entry point — call once per ARKit frame.
    func processARFrame(
        _ arFrame: ARFrame,
        alikedPoints: [TrackedPoint],
        procFrame: UIImage,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        orientation: UIInterfaceOrientation = .landscapeRight
    ) {
        guard isRecording else { return }
        frameCount += 1
        guard frameCount % keyframeInterval == 0 else { return }

        let (fx, fy, cx, cy) = intrinsics
        let arkitUsable = isARKitUsable(arFrame.camera.trackingState)

        if arkitUsable {
            lastGoodARKitViewMatrix = arFrame.camera.viewMatrix(for: orientation)
            trackingSource = .arkit
        } else {
            trackingSource = .fallback
        }

        // ── ArUco detection ───────────────────────────────────────────────
        let aruco = OpenCVBridge.detectArUco5x5(
            procFrame, markerSize: markerSizeMeters,
            procWidth: Int32(procW), procHeight: Int32(procH),
            fx: fx, fy: fy, cx: cx, cy: cy
        )
        let currentArucoR: simd_float3x3? = aruco.detected ? matFromData(aruco.rotationMatrix) : nil
        let currentArucoT: SIMD3<Float>?  = aruco.detected ? vecFromData(aruco.translationVector) : nil

        // ── ALIKED filter ─────────────────────────────────────────────────
        let aliked = alikedPoints.filter { $0.tier == .aliked && !$0.descriptor.isEmpty }
        guard aliked.count >= minAlikedPoints else {
            statusMessage = "Need \(minAlikedPoints) ALIKED pts (\(aliked.count))"
            return
        }

        // ── First keyframe ────────────────────────────────────────────────
        guard !prevKeyframePositions.isEmpty else {
            if arkitUsable {
                storeKeyframe(aliked, viewMatrix: arFrame.camera.viewMatrix(for: orientation),
                              intrinsics: intrinsics, arucoR: currentArucoR, arucoT: currentArucoT)
            }
            return
        }

        // ── Build ordered correspondence arrays ───────────────────────────
        var filteredCommon: [TrackedPoint] = []
        var pts1: [Float] = [], pts2: [Float] = []
        for pt in aliked {
            guard let prev = prevKeyframePositions[pt.id] else { continue }
            pts1.append(prev.x); pts1.append(prev.y)
            pts2.append(pt.position.x); pts2.append(pt.position.y)
            filteredCommon.append(pt)
        }
        let count = filteredCommon.count
        guard count >= 15 else {
            statusMessage = "Too few common pts (\(count))"
            storeKeyframe(aliked, viewMatrix: arkitUsable ? arFrame.camera.viewMatrix(for: orientation) : prevKeyframeViewMatrix,
                          intrinsics: intrinsics, arucoR: currentArucoR, arucoT: currentArucoT)
            return
        }

        // ── Parallax check ────────────────────────────────────────────────
        let parallax = averageParallax(filteredCommon)
        guard parallax >= minParallax else {
            statusMessage = "Move camera (\(String(format: "%.1f", parallax))px parallax)"
            return
        }

        // ── Determine current view matrix ─────────────────────────────────
        guard let currVM = resolveViewMatrix(
            arFrame: arFrame, orientation: orientation,
            arkitUsable: arkitUsable,
            pts1: pts1, pts2: pts2, count: count,
            fx: fx, fy: fy, cx: cx, cy: cy
        ) else { return }

        guard let prevVM = prevKeyframeViewMatrix else { return }

        // ── Triangulate (world-space) ─────────────────────────────────────
        let P1 = projectionFromViewMatrix(prevVM)
        let P2 = projectionFromViewMatrix(currVM)

        var n1: [Float] = [], n2: [Float] = []
        for i in 0..<count {
            n1.append((pts1[i*2]   - cx) / fx); n1.append((pts1[i*2+1] - cy) / fy)
            n2.append((pts2[i*2]   - cx) / fx); n2.append((pts2[i*2+1] - cy) / fy)
        }

        let p1Data = Data(bytes: P1, count: 48)
        let p2Data = Data(bytes: P2, count: 48)
        let n1Data = Data(bytes: n1, count: n1.count * 4)
        let n2Data = Data(bytes: n2, count: n2.count * 4)

        guard let raw = OpenCVBridge.triangulatePoints(
            p1Data, proj2: p2Data, pts1: n1Data, pts2: n2Data, count: count
        ) else {
            statusMessage = "Triangulation failed"
            return
        }

        let worldPts = unpackFloat3(from: raw, count: count)

        // Fallback: apply median-depth normalisation
        let effectivePts: [SIMD3<Float>]
        if trackingSource == .fallback {
            effectivePts = medianScaled(worldPts, prevVM: prevVM)
        } else {
            effectivePts = worldPts
        }

        let (new3D, keptIndices) = filterAndColorWithIndices(
            effectivePts, prevVM: prevVM, currVM: currVM,
            frame: procFrame,
            fx: fx, fy: fy, cx: cx, cy: cy,
            pts1: pts1, pts2: pts2, ptCount: count,
            confidence: trackingSource == .arkit ? 0.6 : 0.3
        )

        // ── Convert to marker-relative and accumulate ─────────────────────
        let worldToMarker = markerToWorldTransform.map { $0.inverse }
        let finalPts = convertToMarkerRelative(new3D, worldToMarker: worldToMarker)
        pointCloud.add(finalPts)
        reconstructedCount = pointCloud.points.count

        // ── Add to visual map (for PnP localization later) ────────────────
        var mapEntries: [VisualMapEntry] = []
        for (j, origIdx) in keptIndices.enumerated() {
            guard origIdx < filteredCommon.count else { continue }
            let desc = filteredCommon[origIdx].descriptor
            guard desc.count == 128 else { continue }
            mapEntries.append(VisualMapEntry(position: finalPts[j].position, descriptor: desc))
        }
        if !mapEntries.isEmpty {
            visualMap.add(mapEntries)
        }

        let src = trackingSource == .arkit ? "ARKit" : "フォールバック"
        statusMessage = "\(reconstructedCount) pts [\(src)] | Map: \(visualMap.count)"

        storeKeyframe(aliked, viewMatrix: currVM,
                      intrinsics: intrinsics, arucoR: currentArucoR, arucoT: currentArucoT)
    }

    // MARK: - Private helpers

    private func convertToMarkerRelative(
        _ points: [Point3D],
        worldToMarker: simd_float4x4?
    ) -> [Point3D] {
        guard let wtm = worldToMarker else { return points }
        return points.map { p in
            let mp = wtm * SIMD4<Float>(p.position, 1)
            return Point3D(position: SIMD3<Float>(mp.x, mp.y, mp.z),
                           color: p.color, confidence: p.confidence)
        }
    }

    private func isARKitUsable(_ state: ARCamera.TrackingState) -> Bool {
        switch state {
        case .normal: return true
        case .limited(let reason):
            switch reason {
            case .excessiveMotion, .initializing: return false
            default: return true
            }
        case .notAvailable: return false
        @unknown default: return false
        }
    }

    private func resolveViewMatrix(
        arFrame: ARFrame,
        orientation: UIInterfaceOrientation,
        arkitUsable: Bool,
        pts1: [Float], pts2: [Float], count: Int,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> simd_float4x4? {

        if arkitUsable {
            return arFrame.camera.viewMatrix(for: orientation)
        }

        guard let prevVM = prevKeyframeViewMatrix else {
            statusMessage = "No anchor for fallback"
            return nil
        }
        let pts1Data = Data(bytes: pts1, count: pts1.count * 4)
        let pts2Data = Data(bytes: pts2, count: pts2.count * 4)
        let pose = OpenCVBridge.recoverPose(
            pts1: pts1Data, pts2: pts2Data, count: count,
            fx: fx, fy: fy, cx: cx, cy: cy
        )
        guard pose.success, pose.inlierCount >= 15 else {
            statusMessage = "Pose failed (fallback, \(pose.inlierCount) inliers)"
            return nil
        }
        let R_rel = matFromData(pose.rotationMatrix)
        let t_rel = vecFromData(pose.translationVector)

        let R_prev = simd_float3x3(prevVM.columns.0.xyz, prevVM.columns.1.xyz, prevVM.columns.2.xyz)
        let t_prev = prevVM.columns.3.xyz
        let R_curr = R_rel * R_prev
        let t_curr = R_rel * t_prev + t_rel
        return simd_float4x4(
            SIMD4<Float>(R_curr.columns.0, 0),
            SIMD4<Float>(R_curr.columns.1, 0),
            SIMD4<Float>(R_curr.columns.2, 0),
            SIMD4<Float>(t_curr, 1)
        )
    }

    private func medianScaled(_ pts: [SIMD3<Float>], prevVM: simd_float4x4) -> [SIMD3<Float>] {
        let depths = pts.compactMap { p -> Float? in
            guard p != .zero else { return nil }
            let camPt = prevVM * SIMD4<Float>(p, 1)
            return camPt.z > 0.01 ? camPt.z : nil
        }.sorted()
        guard !depths.isEmpty else { return pts }
        let median = depths[depths.count / 2]
        guard median > 0.001 else { return pts }
        let scale = 1.0 / median
        return pts.map { $0 == .zero ? .zero : $0 * scale }
    }

    /// Reprojection-error filter + color sampling, also returns indices of kept points.
    private func filterAndColorWithIndices(
        _ worldPts: [SIMD3<Float>],
        prevVM: simd_float4x4,
        currVM: simd_float4x4,
        frame: UIImage,
        fx: Float, fy: Float, cx: Float, cy: Float,
        pts1: [Float], pts2: [Float], ptCount: Int,
        confidence: Float
    ) -> (points: [Point3D], keptIndices: [Int]) {
        let maxErr: Float  = 2.0
        let minZ: Float    = 0.01
        let maxZ: Float    = 20.0
        var result: [Point3D] = []
        var kept: [Int] = []

        for (i, wp) in worldPts.enumerated() {
            guard wp != .zero else { continue }

            let c1 = prevVM * SIMD4<Float>(wp, 1)
            guard c1.z > minZ && c1.z < maxZ else { continue }
            let e1x = (c1.x / c1.z) * fx + cx - (i < ptCount ? pts1[i*2]   : cx)
            let e1y = (c1.y / c1.z) * fy + cy - (i < ptCount ? pts1[i*2+1] : cy)
            guard sqrt(e1x*e1x + e1y*e1y) < maxErr else { continue }

            let c2 = currVM * SIMD4<Float>(wp, 1)
            guard c2.z > 0.001 else { continue }
            let e2x = (c2.x / c2.z) * fx + cx - (i < ptCount ? pts2[i*2]   : cx)
            let e2y = (c2.y / c2.z) * fy + cy - (i < ptCount ? pts2[i*2+1] : cy)
            guard sqrt(e2x*e2x + e2y*e2y) < maxErr else { continue }

            let color = sampleColor(frame, x: i < ptCount ? pts2[i*2] : procW/2,
                                           y: i < ptCount ? pts2[i*2+1] : procH/2)
            result.append(Point3D(position: wp, color: color, confidence: confidence))
            kept.append(i)
        }
        return (result, kept)
    }

    // MARK: - Geometry utilities

    private func projectionFromViewMatrix(_ vm: simd_float4x4) -> [Float] {
        var P = [Float](repeating: 0, count: 12)
        P[0]  = vm.columns.0.x; P[1]  = vm.columns.1.x; P[2]  = vm.columns.2.x; P[3]  = vm.columns.3.x
        P[4]  = vm.columns.0.y; P[5]  = vm.columns.1.y; P[6]  = vm.columns.2.y; P[7]  = vm.columns.3.y
        P[8]  = vm.columns.0.z; P[9]  = vm.columns.1.z; P[10] = vm.columns.2.z; P[11] = vm.columns.3.z
        return P
    }

    private func matFromData(_ data: Data) -> simd_float3x3 {
        guard data.count >= 36 else { return matrix_identity_float3x3 }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return simd_float3x3(
            SIMD3<Float>(f[0], f[3], f[6]),
            SIMD3<Float>(f[1], f[4], f[7]),
            SIMD3<Float>(f[2], f[5], f[8])
        )
    }

    private func vecFromData(_ data: Data) -> SIMD3<Float> {
        guard data.count >= 12 else { return .zero }
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return SIMD3<Float>(f[0], f[1], f[2])
    }

    private func unpackFloat3(from data: Data, count: Int) -> [SIMD3<Float>] {
        let f = data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        return (0..<count).map { i in SIMD3<Float>(f[i*3], f[i*3+1], f[i*3+2]) }
    }

    private func averageParallax(_ points: [TrackedPoint]) -> Float {
        var total: Float = 0; var n = 0
        for pt in points {
            guard let prev = prevKeyframePositions[pt.id] else { continue }
            let d = pt.position - prev
            total += sqrt(d.x*d.x + d.y*d.y); n += 1
        }
        return n > 0 ? total / Float(n) : 0
    }

    private func storeKeyframe(
        _ points: [TrackedPoint],
        viewMatrix: simd_float4x4?,
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        arucoR: simd_float3x3?,
        arucoT: SIMD3<Float>?
    ) {
        prevKeyframePositions   = Dictionary(uniqueKeysWithValues: points.map { ($0.id, $0.position) })
        prevKeyframeDescriptors = Dictionary(uniqueKeysWithValues: points.map { ($0.id, $0.descriptor) })
        prevKeyframeViewMatrix  = viewMatrix
        prevKeyframeIntrinsics  = intrinsics
        prevKeyframeArucoR      = arucoR
        prevKeyframeArucoT      = arucoT
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
        guard let ctx = CGContext(data: &pixel, width: 1, height: 1,
                                  bitsPerComponent: 8, bytesPerRow: 4,
                                  space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return SIMD3<Float>(0.5, 0.5, 0.5) }
        ctx.draw(cg, in: CGRect(x: -CGFloat(px), y: -CGFloat(py), width: iw, height: ih))
        return SIMD3<Float>(Float(pixel[0])/255, Float(pixel[1])/255, Float(pixel[2])/255)
    }
}

// MARK: - SIMD helpers

private extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

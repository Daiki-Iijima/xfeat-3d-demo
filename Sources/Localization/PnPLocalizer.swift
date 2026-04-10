import Foundation
import simd

/// Self-localizer using descriptor matching + solvePnPRansac.
///
/// After ARKit is disabled, call `localize(trackedPoints:intrinsics:visualMap:)`
/// every frame to get the current camera pose in marker-relative space.
@MainActor
final class PnPLocalizer: ObservableObject {

    /// Estimated camera-to-world transform in marker-relative space.
    /// `columns.3.xyz` is the camera position relative to the marker origin.
    @Published private(set) var estimatedPose: simd_float4x4? = nil

    /// Minimum number of inliers required to accept a PnP result.
    var minInliers: Int = 8

    // MARK: - Localization

    /// Attempt to localize the current frame against the visual map.
    ///
    /// - Parameters:
    ///   - trackedPoints: All tracked points (uses ALIKED-tier only).
    ///   - intrinsics: Camera intrinsics at proc resolution.
    ///   - visualMap: The built visual map.
    func localize(
        trackedPoints: [TrackedPoint],
        intrinsics: (fx: Float, fy: Float, cx: Float, cy: Float),
        visualMap: VisualMap
    ) {
        guard !visualMap.isEmpty else { estimatedPose = nil; return }

        let aliked = trackedPoints.filter { $0.tier == .aliked && $0.descriptor.count == 128 }
        guard aliked.count >= 6 else { estimatedPose = nil; return }

        let descDim = 128
        let queryCount = aliked.count

        // Build flat query descriptor matrix
        var queryDescs = [Float](repeating: 0, count: queryCount * descDim)
        for (i, pt) in aliked.enumerated() {
            let base = i * descDim
            for j in 0..<descDim {
                queryDescs[base + j] = pt.descriptor[j]
            }
        }

        let matches = visualMap.findMatches(
            queryDescriptors: queryDescs,
            queryCount: queryCount,
            descDim: descDim,
            minScore: 0.72
        )
        guard matches.count >= 6 else { estimatedPose = nil; return }

        // Build 3D-2D correspondence buffers
        var pts3D = [Float]()
        var pts2D = [Float]()
        pts3D.reserveCapacity(matches.count * 3)
        pts2D.reserveCapacity(matches.count * 2)

        for match in matches {
            let mp = visualMap.entries[match.entryIdx]
            let pt = aliked[match.queryIdx]
            pts3D.append(mp.position.x); pts3D.append(mp.position.y); pts3D.append(mp.position.z)
            pts2D.append(pt.position.x); pts2D.append(pt.position.y)
        }

        let result = OpenCVBridge.solvePnP(
            points3D: Data(bytes: pts3D, count: pts3D.count * 4),
            points2D: Data(bytes: pts2D, count: pts2D.count * 4),
            count:    matches.count,
            fx: intrinsics.fx, fy: intrinsics.fy,
            cx: intrinsics.cx, cy: intrinsics.cy
        )

        guard result.success, result.inlierCount >= minInliers else {
            estimatedPose = nil
            return
        }

        // result gives world-to-camera (marker space) — invert to camera-to-world
        let vm = viewMatrix(fromR: result.rotationMatrix, t: result.translationVector)
        estimatedPose = vm.inverse
    }

    func reset() {
        estimatedPose = nil
    }

    // MARK: - Helpers

    private func viewMatrix(fromR rData: Data, t tData: Data) -> simd_float4x4 {
        guard rData.count >= 36, tData.count >= 12 else { return matrix_identity_float4x4 }
        let rf = rData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        let tf = tData.withUnsafeBytes { $0.bindMemory(to: Float.self) }
        // Row-major 3×3 in data → simd column-major
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
}

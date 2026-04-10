import simd

/// A single 3D reconstructed point.
struct Point3D: Sendable {
    /// 3D position in world coordinates (metres when ArUco scale is calibrated).
    var position: SIMD3<Float>
    /// RGB color sampled from the camera frame [0,1].
    var color: SIMD3<Float>
    /// Reconstruction confidence: 0 (low) to 1 (high).
    var confidence: Float
}

/// Accumulated 3D point cloud with O(1) spatial deduplication via hash grid.
final class PointCloud: @unchecked Sendable {

    private(set) var points: [Point3D] = []

    /// Merge-radius in world units (metres when scale is calibrated, arbitrary otherwise).
    var mergeRadius: Float = 0.02

    /// Maximum cloud size (memory/performance cap).
    var maxPoints: Int = 50_000

    // Spatial hash: cell key → indices into `points`
    private var spatialHash: [SIMD3<Int>: [Int]] = [:]

    // MARK: - Public API

    /// Add new 3D points, merging near-duplicates in O(1) amortised time.
    func add(_ newPoints: [Point3D]) {
        for p in newPoints {
            if let idx = nearestIndex(to: p.position) {
                // Merge: blend position and color, boost confidence.
                let old = points[idx]
                let oldKey = cellKey(for: old.position)
                let w = old.confidence / (old.confidence + p.confidence)
                let merged = Point3D(
                    position:   old.position * w + p.position * (1 - w),
                    color:      old.color    * w + p.color    * (1 - w),
                    confidence: min(1.0, old.confidence + p.confidence * 0.1)
                )
                points[idx] = merged
                // Update hash if the merged position crossed a cell boundary
                let newKey = cellKey(for: merged.position)
                if newKey != oldKey {
                    spatialHash[oldKey]?.removeAll { $0 == idx }
                    spatialHash[newKey, default: []].append(idx)
                }
            } else {
                guard points.count < maxPoints else { continue }
                let key = cellKey(for: p.position)
                spatialHash[key, default: []].append(points.count)
                points.append(p)
            }
        }
    }

    func clear() {
        points = []
        spatialHash = [:]
    }

    // MARK: - Private

    /// Converts a world position to its cell key in the hash grid.
    private func cellKey(for pos: SIMD3<Float>) -> SIMD3<Int> {
        let inv = 1.0 / mergeRadius
        return SIMD3<Int>(
            Int(floor(pos.x * inv)),
            Int(floor(pos.y * inv)),
            Int(floor(pos.z * inv))
        )
    }

    /// Returns the index of the nearest point within `mergeRadius`, or nil.
    /// Searches only the 27 neighbouring cells — O(1) amortised.
    private func nearestIndex(to pos: SIMD3<Float>) -> Int? {
        let key = cellKey(for: pos)
        let r2 = mergeRadius * mergeRadius
        for dx in -1...1 {
            for dy in -1...1 {
                for dz in -1...1 {
                    let nk = SIMD3<Int>(key.x + dx, key.y + dy, key.z + dz)
                    guard let bucket = spatialHash[nk] else { continue }
                    for i in bucket {
                        let d = points[i].position - pos
                        if simd_dot(d, d) < r2 { return i }
                    }
                }
            }
        }
        return nil
    }
}

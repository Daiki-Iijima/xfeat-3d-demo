import Foundation

/// A feature point with a persistent identity across frames.
struct TrackedPoint: Identifiable, Sendable {
    /// Monotonically increasing ID assigned when the point is first detected.
    let id: Int
    /// Current position in procSize coordinate space.
    var position: SIMD2<Float>
    /// Number of frames this point has been alive.
    var age: Int
    /// Latest L2-normalised descriptor. Empty for ORB-tier points.
    var descriptor: [Float]
    /// Which extractor last confirmed this point. Can only increase (promotion only).
    var tier: TrackTier
}

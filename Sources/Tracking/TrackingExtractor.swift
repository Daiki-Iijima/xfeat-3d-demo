/// Feature extractor strategy for tracking mode.
enum TrackingExtractor: String, CaseIterable, Identifiable {
    /// Single-extractor modes — use one model for both detection and descriptor.
    case xfeat      = "XFeat"
    case aliked     = "ALIKED"
    case superpoint = "SuperPoint"
    /// Three-tier hierarchical mode:
    /// ORB (frequent, position-only) → XFeat (medium interval, descriptor) → ALIKED (slow, highest quality).
    /// Higher tiers promote matching lower-tier tracks in-place, preserving IDs.
    case hierarchical = "階層"
    var id: String { rawValue }
}

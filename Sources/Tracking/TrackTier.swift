/// Confidence tier of a tracked point — determined by which extractor
/// detected or last confirmed it. Points can only be promoted, never demoted.
enum TrackTier: Int, Sendable, Comparable {
    case orb    = 0  // ORB: fast broad coverage, no descriptor
    case xfeat  = 1  // XFeat: medium quality, 64-dim L2-normalised descriptor
    case aliked = 2  // ALIKED: highest quality, 128-dim L2-normalised descriptor

    static func < (lhs: TrackTier, rhs: TrackTier) -> Bool { lhs.rawValue < rhs.rawValue }

    var label: String {
        switch self {
        case .orb:    return "ORB"
        case .xfeat:  return "XFeat"
        case .aliked: return "ALIKED"
        }
    }
}

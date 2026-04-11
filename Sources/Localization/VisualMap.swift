import Foundation
import simd
import Accelerate

/// A single 3D landmark in the visual map, paired with its descriptor.
struct VisualMapEntry: Sendable {
    /// 3D position in marker-relative coordinate space (metres).
    let position: SIMD3<Float>
    /// ALIKED descriptor (128D, L2-normalised).
    let descriptor: [Float]
}

// MARK: - Map culling state (parallel arrays, VisualMap-private)
// observationCounts[i]: how many times entry i has been matched
// lastSeenFrames[i]:    frame index of the most recent match

/// Visual map for descriptor-based self-localization (PnP).
///
/// Built while ARKit is running: triangulated 3D points and their ALIKED
/// descriptors are added via `add(_:)`.  After ARKit is disabled,
/// `findMatches(queryDescriptors:queryCount:descDim:minScore:)` returns
/// 2D-index → 3D-index correspondences for solvePnP.
final class VisualMap: @unchecked Sendable {

    private(set) var entries: [VisualMapEntry] = []
    private var observationCounts: [Int] = []
    private var lastSeenFrames:    [Int] = []

    // Stable-ID tracking — survives culling so keyframe references stay valid.
    private var entryIDs: [Int] = []          // stable ID for entries[i]
    private var nextID:   Int   = 0
    /// Stable-ID → current array index. Rebuilt after every cull().
    private(set) var idToIndex: [Int: Int] = [:]

    var isEmpty: Bool { entries.isEmpty }
    var count:   Int  { entries.count }

    // MARK: - Building

    func add(_ newEntries: [VisualMapEntry], currentFrame: Int = 0) {
        let startIdx = entries.count
        entries.append(contentsOf: newEntries)
        observationCounts.append(contentsOf: Array(repeating: 1, count: newEntries.count))
        lastSeenFrames.append(contentsOf: Array(repeating: currentFrame, count: newEntries.count))
        for i in 0..<newEntries.count {
            let id = nextID
            nextID += 1
            entryIDs.append(id)
            idToIndex[id] = startIdx + i
        }
    }

    func clear() {
        entries = []
        observationCounts = []
        lastSeenFrames    = []
        entryIDs          = []
        idToIndex         = [:]
    }

    // MARK: - Stable-ID accessors

    /// Returns the stable ID for the entry at the given array index.
    func id(at index: Int) -> Int {
        entryIDs[index]
    }

    /// Returns the current array index for a stable ID, or nil if culled.
    func index(forID id: Int) -> Int? {
        idToIndex[id]
    }

    /// Returns the entry for a stable ID, or nil if culled.
    func entry(forID id: Int) -> VisualMapEntry? {
        guard let idx = idToIndex[id] else { return nil }
        return entries[idx]
    }

    /// Update the 3D position of an entry by stable ID (used after Bundle Adjustment).
    func updatePosition(id: Int, newPosition: SIMD3<Float>) {
        guard let idx = idToIndex[id] else { return }
        let old = entries[idx]
        entries[idx] = VisualMapEntry(position: newPosition, descriptor: old.descriptor)
    }

    /// Returns a stable-ID → 3D position dictionary suitable for use in background tasks.
    /// The snapshot is value-copied, so it remains valid even if the map is later mutated.
    func idToPositionSnapshot() -> [Int: SIMD3<Float>] {
        var dict = [Int: SIMD3<Float>]()
        dict.reserveCapacity(entries.count)
        for (id, idx) in idToIndex {
            dict[id] = entries[idx].position
        }
        return dict
    }

    // MARK: - Observation tracking

    /// Call after each successful match to keep per-entry stats current.
    func markObserved(indices: [Int], frame: Int) {
        for i in indices {
            guard i < entries.count else { continue }
            observationCounts[i] += 1
            lastSeenFrames[i] = frame
        }
    }

    // MARK: - Culling

    /// Remove stale map points.
    ///
    /// A point is removed when **both** conditions hold:
    /// - Not seen for `maxUnseenFrames` frames (stale).
    /// - Observed fewer than `minObservations` times total (unreliable).
    ///
    /// Points that have been matched often enough are treated as stable landmarks
    /// and are kept even when temporarily out of view.
    ///
    /// - Returns: Number of points removed.
    @discardableResult
    func cull(currentFrame: Int,
              maxUnseenFrames: Int = 150,
              minObservations: Int = 4) -> Int {
        var keep: [Int] = []
        keep.reserveCapacity(entries.count)
        for i in 0..<entries.count {
            let stale    = (currentFrame - lastSeenFrames[i]) > maxUnseenFrames
            let reliable = observationCounts[i] >= minObservations
            if !stale || reliable { keep.append(i) }
        }
        let removed = entries.count - keep.count
        guard removed > 0 else { return 0 }
        entries           = keep.map { entries[$0] }
        observationCounts = keep.map { observationCounts[$0] }
        lastSeenFrames    = keep.map { lastSeenFrames[$0] }
        entryIDs          = keep.map { entryIDs[$0] }
        // Rebuild the ID→index lookup after reordering
        idToIndex = [:]
        for (newIdx, id) in entryIDs.enumerated() {
            idToIndex[id] = newIdx
        }
        return removed
    }

    // MARK: - Matching

    /// Find best matching map entries for a set of query descriptors using mutual NN.
    ///
    /// - Parameters:
    ///   - queryDescriptors: Flat row-major Float array (queryCount × descDim).
    ///   - queryCount: Number of query descriptors.
    ///   - descDim: Descriptor dimension (e.g. 128 for ALIKED).
    ///   - minScore: Minimum cosine similarity to accept a match.
    /// - Returns: Array of (entryIdx, queryIdx) pairs sorted by score descending.
    func findMatches(
        queryDescriptors: [Float],
        queryCount: Int,
        descDim: Int,
        minScore: Float = 0.72
    ) -> [(entryIdx: Int, queryIdx: Int)] {
        VisualMap.computeMatches(
            queryDescriptors: queryDescriptors, queryCount: queryCount,
            descDim: descDim, minScore: minScore, entries: entries)
    }

    /// Nonisolated static version for background execution.
    /// Identical logic to `findMatches` but takes `entries` as a value-type parameter
    /// so the caller can snapshot `self.entries` and dispatch to `Task.detached`.
    nonisolated static func computeMatches(
        queryDescriptors: [Float],
        queryCount: Int,
        descDim: Int,
        minScore: Float = 0.72,
        entries: [VisualMapEntry]
    ) -> [(entryIdx: Int, queryIdx: Int)] {
        guard !entries.isEmpty, queryCount > 0, descDim > 0 else { return [] }

        let mapCount = entries.count

        // Build flat map descriptor matrix (mapCount × descDim)
        var mapDescs = [Float](repeating: 0, count: mapCount * descDim)
        for (i, entry) in entries.enumerated() {
            guard entry.descriptor.count == descDim else { continue }
            let base = i * descDim
            for j in 0..<descDim {
                mapDescs[base + j] = entry.descriptor[j]
            }
        }

        // Compute cosine similarity matrix S[queryCount × mapCount]
        // S[qi * mapCount + mi] = cosine_sim(query[qi], map[mi])
        var S = [Float](repeating: 0, count: queryCount * mapCount)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            Int32(queryCount), Int32(mapCount), Int32(descDim),
            1.0,
            queryDescriptors, Int32(descDim),
            mapDescs,         Int32(descDim),
            0.0,
            &S, Int32(mapCount)
        )

        // Mutual nearest-neighbour matching
        var results: [(entryIdx: Int, queryIdx: Int, score: Float)] = []

        for qi in 0..<queryCount {
            let rowBase = qi * mapCount
            // Best map entry for this query
            var bestScore: Float = minScore
            var bestMi = -1
            for mi in 0..<mapCount {
                let s = S[rowBase + mi]
                if s > bestScore { bestScore = s; bestMi = mi }
            }
            guard bestMi >= 0 else { continue }

            // Verify mutual: best query for bestMi
            var bestQScore: Float = -1
            var bestQi = -1
            for qi2 in 0..<queryCount {
                let s = S[qi2 * mapCount + bestMi]
                if s > bestQScore { bestQScore = s; bestQi = qi2 }
            }
            if bestQi == qi {
                results.append((entryIdx: bestMi, queryIdx: qi, score: bestScore))
            }
        }

        results.sort { $0.score > $1.score }
        return results.map { (entryIdx: $0.entryIdx, queryIdx: $0.queryIdx) }
    }
}

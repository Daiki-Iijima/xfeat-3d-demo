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

/// Visual map for descriptor-based self-localization (PnP).
///
/// Built while ARKit is running: triangulated 3D points and their ALIKED
/// descriptors are added via `add(_:)`.  After ARKit is disabled,
/// `findMatches(queryDescriptors:queryCount:descDim:minScore:)` returns
/// 2D-index → 3D-index correspondences for solvePnP.
final class VisualMap: @unchecked Sendable {

    private(set) var entries: [VisualMapEntry] = []

    var isEmpty: Bool { entries.isEmpty }
    var count:   Int  { entries.count }

    // MARK: - Building

    func add(_ newEntries: [VisualMapEntry]) {
        entries.append(contentsOf: newEntries)
    }

    func clear() {
        entries = []
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

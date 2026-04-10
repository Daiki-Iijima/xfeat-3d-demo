import SwiftUI
import SceneKit

// MARK: - Color Filter

enum PointColorFilter: String, CaseIterable {
    case all   = "全て"
    case red   = "赤"
    case green = "緑"
    case blue  = "青"

    var icon: String {
        switch self {
        case .all:   return "circle.grid.3x3"
        case .red:   return "circle.fill"
        case .green: return "circle.fill"
        case .blue:  return "circle.fill"
        }
    }

    var tint: Color {
        switch self {
        case .all:   return .white
        case .red:   return .red
        case .green: return .green
        case .blue:  return .blue
        }
    }

    /// Returns true if the given RGB color matches this filter.
    func matches(_ c: SIMD3<Float>) -> Bool {
        switch self {
        case .all:   return true
        case .red:   return c.x > 0.35 && c.x > c.y * 1.25 && c.x > c.z * 1.25
        case .green: return c.y > 0.30 && c.y > c.x * 1.15 && c.y > c.z * 1.15
        case .blue:  return c.z > 0.30 && c.z > c.x * 1.15 && c.z > c.y * 1.15
        }
    }
}

// MARK: - PointCloudView

/// SceneKit view that renders the accumulated 3D point cloud.
struct PointCloudView: UIViewRepresentable {
    let points: [Point3D]
    var colorFilter: PointColorFilter = .all
    /// Called with the nearest visible Point3D when the user taps the view.
    var onPointTapped: ((Point3D) -> Void)? = nil

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.backgroundColor = UIColor(white: 0.05, alpha: 1)
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = false
        scnView.showsStatistics = false

        let scene = SCNScene()
        scnView.scene = scene

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 0, 3)
        scene.rootNode.addChildNode(cameraNode)

        // Ambient light
        let ambientNode = SCNNode()
        ambientNode.light = SCNLight()
        ambientNode.light?.type = .ambient
        ambientNode.light?.intensity = 1000
        scene.rootNode.addChildNode(ambientNode)

        // Axes helper
        addAxes(to: scene.rootNode)

        // Point cloud node (placeholder)
        let cloudNode = SCNNode()
        cloudNode.name = "pointCloud"
        scene.rootNode.addChildNode(cloudNode)

        context.coordinator.scnView = scnView
        context.coordinator.cloudNode = cloudNode

        // Tap gesture for point inspection
        let tap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        scnView.addGestureRecognizer(tap)

        return scnView
    }

    func updateUIView(_ scnView: SCNView, context: Context) {
        context.coordinator.onPointTapped = onPointTapped
        context.coordinator.updatePoints(points, filter: colorFilter)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    // MARK: - Coordinator

    final class Coordinator {
        weak var scnView: SCNView?
        weak var cloudNode: SCNNode?
        var onPointTapped: ((Point3D) -> Void)? = nil

        /// Filtered points currently rendered — used for tap hit detection.
        private var displayedPoints: [Point3D] = []
        private var lastCount: Int = -1
        private var lastFilter: PointColorFilter = .all

        // MARK: Update geometry

        func updatePoints(_ points: [Point3D], filter: PointColorFilter) {
            let filtered = filter == .all ? points : points.filter { filter.matches($0.color) }
            guard filtered.count != lastCount || filter != lastFilter,
                  let cloudNode else { return }
            lastCount    = filtered.count
            lastFilter   = filter
            displayedPoints = filtered

            guard !filtered.isEmpty else {
                cloudNode.geometry = nil
                return
            }

            // Remove previous geometry
            cloudNode.geometry = nil

            // Vertex positions
            let vertices = filtered.map { SCNVector3($0.position.x, $0.position.y, $0.position.z) }
            let vertexSource = SCNGeometrySource(vertices: vertices)

            // Per-vertex colors (RGBA float32)
            var colorData = [Float](repeating: 0, count: filtered.count * 4)
            for (i, p) in filtered.enumerated() {
                colorData[i * 4]     = p.color.x
                colorData[i * 4 + 1] = p.color.y
                colorData[i * 4 + 2] = p.color.z
                colorData[i * 4 + 3] = 1.0
            }
            let colorNSData = Data(bytes: colorData, count: colorData.count * 4)
            let colorSource = SCNGeometrySource(
                data: colorNSData,
                semantic: .color,
                vectorCount: filtered.count,
                usesFloatComponents: true,
                componentsPerVector: 4,
                bytesPerComponent: 4,
                dataOffset: 0,
                dataStride: 16
            )

            // Point indices
            let indices = (0..<filtered.count).map { Int32($0) }
            let indexData = Data(bytes: indices, count: indices.count * 4)
            let element = SCNGeometryElement(
                data: indexData,
                primitiveType: .point,
                primitiveCount: filtered.count,
                bytesPerIndex: 4
            )
            element.pointSize = 3.0
            element.minimumPointScreenSpaceRadius = 1.0
            element.maximumPointScreenSpaceRadius = 5.0

            let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.lightingModel = .constant
            geometry.materials = [material]

            cloudNode.geometry = geometry
        }

        // MARK: Tap handling

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let scnView else { return }
            let tapLocation = gesture.location(in: scnView)

            var bestDist: CGFloat = 44   // max tap radius in points
            var bestPoint: Point3D? = nil

            for point in displayedPoints {
                let worldPos = SCNVector3(point.position.x, point.position.y, point.position.z)
                let projected = scnView.projectPoint(worldPos)
                // z in [0,1] — values outside are behind camera or past far clip
                guard projected.z >= 0, projected.z <= 1 else { continue }
                let dx = CGFloat(projected.x) - tapLocation.x
                let dy = CGFloat(projected.y) - tapLocation.y
                let dist = sqrt(dx * dx + dy * dy)
                if dist < bestDist {
                    bestDist = dist
                    bestPoint = point
                }
            }

            if let best = bestPoint {
                onPointTapped?(best)
            }
        }
    }

    // MARK: - Axes Helper

    private func addAxes(to node: SCNNode) {
        let axisLength: Float = 0.1
        let items: [(UIColor, SIMD3<Float>)] = [
            (.red,   SIMD3<Float>(axisLength, 0, 0)),
            (.green, SIMD3<Float>(0, axisLength, 0)),
            (.blue,  SIMD3<Float>(0, 0, axisLength))
        ]
        for (color, dir) in items {
            let cyl = SCNCylinder(radius: 0.002, height: CGFloat(axisLength))
            cyl.firstMaterial?.diffuse.contents = color
            cyl.firstMaterial?.lightingModel = .constant
            let axNode = SCNNode(geometry: cyl)
            axNode.position = SCNVector3(dir.x / 2, dir.y / 2, dir.z / 2)
            if dir.x > 0 { axNode.eulerAngles = SCNVector3(0, 0, -Float.pi / 2) }
            if dir.z > 0 { axNode.eulerAngles = SCNVector3(Float.pi / 2, 0, 0) }
            node.addChildNode(axNode)
        }
    }
}

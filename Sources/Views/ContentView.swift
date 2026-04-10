import SwiftUI
import ARKit

struct ContentView: View {
    @StateObject private var arManager = ARSessionManager()

    var body: some View {
        NavigationStack {
            VOModeView(arManager: arManager)
        }
        .onAppear {
            arManager.start()
            MotionManager.shared.start()
        }
        .onDisappear {
            arManager.pause()
            MotionManager.shared.stop()
        }
    }
}

// MARK: - ARKit Camera View

struct ARCameraView: UIViewRepresentable {
    let arSession: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session = arSession
        view.automaticallyUpdatesLighting = false
        view.scene = SCNScene()
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

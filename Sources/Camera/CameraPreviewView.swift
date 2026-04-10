import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = cameraManager.captureSession
        view.previewLayer.videoGravity = .resizeAspectFill
        applyRotation(cameraManager.videoRotationAngle, to: view)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        applyRotation(cameraManager.videoRotationAngle, to: uiView)
    }

    private func applyRotation(_ angle: CGFloat, to view: PreviewUIView) {
        guard let conn = view.previewLayer.connection else { return }
        if #available(iOS 17.0, *) {
            if conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
    }

    class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

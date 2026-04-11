import Foundation
import ARKit
import UIKit
import CoreImage
import simd

/// ARKit session wrapper.
///
/// Replaces CameraManager as the source of camera frames and pose.
/// Converts each ARFrame's pixel buffer to a proc-resolution UIImage for
/// feature extraction, and publishes the ARKit tracking state so downstream
/// code can handle limited/unavailable states gracefully.
@MainActor
final class ARSessionManager: NSObject, ObservableObject {

    let arSession = ARSession()

    /// Latest raw ARKit frame (includes pose + intrinsics).
    @Published var latestFrame: ARFrame?
    /// Camera image scaled to procW × procH for feature extraction.
    @Published var procImage: UIImage?
    /// Current ARKit tracking state.
    @Published var trackingState: ARCamera.TrackingState = .notAvailable
    @Published var isRunning = false

    /// Marker-to-world transform set once when the ArUco marker is confirmed.
    /// Used to express all positions relative to the marker origin (0,0,0).
    @Published var markerToWorldTransform: simd_float4x4? = nil

    /// Processing resolution used throughout the pipeline.
    let procW: Float = 960
    let procH: Float = 720

    // Thread-safe: CIContext is documented as safe for concurrent rendering.
    nonisolated(unsafe) private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Frame mailbox (prevents ARFrame accumulation)
    //
    // ARKit delivers frames at ~60 fps on a background thread. If the MainActor is
    // busy (feature extraction), queued `Task { @MainActor in }` closures each retain
    // their ARFrame, triggering ARKit's "retaining N ARFrames" warning and eventually
    // stalling the camera feed.
    //
    // Solution: mailbox pattern — only the *latest* frame is kept. A single MainActor
    // task drains it; while that task is pending no new task is scheduled.
    nonisolated(unsafe) private let mailboxLock = NSLock()
    nonisolated(unsafe) private var mailboxFrame: ARFrame?
    nonisolated(unsafe) private var mailboxImage: UIImage?
    nonisolated(unsafe) private var mailboxPending = false

    override init() {
        super.init()
        arSession.delegate = self
    }

    // MARK: - Session lifecycle

    func start() {
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        config.environmentTexturing = .none
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
    }

    func pause() {
        arSession.pause()
        isRunning = false
    }

    // MARK: - Marker origin

    /// Set the marker-to-world transform from an ArUco detection result.
    ///
    /// Call this once when the marker is confidently detected during calibration
    /// and ARKit tracking is normal.  All positions will then be expressed relative
    /// to the marker's coordinate frame.
    ///
    /// - Parameters:
    ///   - arucoRotation:    9 float32 bytes, row-major 3×3 rotation (marker→camera).
    ///   - arucoTranslation: 3 float32 bytes, marker centre in camera frame (metres).
    ///   - arFrame:          The current ARKit frame used to get camera-to-world.
    ///   - orientation:      Interface orientation for viewMatrix selection.
    func setMarkerOrigin(
        arucoRotation: Data,
        arucoTranslation: Data,
        arFrame: ARFrame,
        orientation: UIInterfaceOrientation = .landscapeRight
    ) {
        guard arucoRotation.count >= 36, arucoTranslation.count >= 12 else { return }

        let rf = arucoRotation.withUnsafeBytes    { $0.bindMemory(to: Float.self) }
        let tf = arucoTranslation.withUnsafeBytes { $0.bindMemory(to: Float.self) }

        // Row-major 3×3 in data → simd column-major simd_float3x3
        let R = simd_float3x3(
            SIMD3<Float>(rf[0], rf[3], rf[6]),
            SIMD3<Float>(rf[1], rf[4], rf[7]),
            SIMD3<Float>(rf[2], rf[5], rf[8])
        )
        let t = SIMD3<Float>(tf[0], tf[1], tf[2])

        // Marker-to-camera homogeneous transform
        let T_CM = simd_float4x4(
            SIMD4<Float>(R.columns.0, 0),
            SIMD4<Float>(R.columns.1, 0),
            SIMD4<Float>(R.columns.2, 0),
            SIMD4<Float>(t, 1)
        )

        // camera-to-world from ARKit
        let T_WC = arFrame.camera.viewMatrix(for: orientation).inverse

        // marker-to-world
        markerToWorldTransform = T_WC * T_CM
    }

    // MARK: - Helpers

    /// Scaled camera intrinsics for the proc resolution from a given ARFrame.
    func scaledIntrinsics(for frame: ARFrame) -> (fx: Float, fy: Float, cx: Float, cy: Float) {
        let size = frame.camera.imageResolution
        // ARCamera.intrinsics is column-major simd_float3x3
        // intr[col][row]: fx = [0][0], fy = [1][1], cx = [2][0], cy = [2][1]
        let intr = frame.camera.intrinsics
        let sx = procW / Float(size.width)
        let sy = procH / Float(size.height)
        return (fx: intr[0][0] * sx, fy: intr[1][1] * sy,
                cx: intr[2][0] * sx, cy: intr[2][1] * sy)
    }

    /// True when the ARKit pose should be trusted for reconstruction.
    nonisolated func isUsable(_ state: ARCamera.TrackingState) -> Bool {
        switch state {
        case .normal: return true
        case .limited(let reason):
            switch reason {
            case .excessiveMotion, .initializing: return false
            default: return true   // relocalizing / insufficientFeatures — pose still available
            }
        case .notAvailable: return false
        @unknown default: return false
        }
    }

    // MARK: - CVPixelBuffer → UIImage

    nonisolated private func makeUIImage(from pixelBuffer: CVPixelBuffer) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaleX = CGFloat(procW) / w
        let scaleY = CGFloat(procH) / h
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        guard let cg = ciContext.createCGImage(scaled, from: scaled.extent) else { return nil }
        return UIImage(cgImage: cg)
    }
}

// MARK: - ARSessionDelegate

extension ARSessionManager: ARSessionDelegate {

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let image = makeUIImage(from: frame.capturedImage)

        // Deposit the latest frame into the mailbox (replaces any unseen frame).
        mailboxLock.lock()
        mailboxFrame = frame
        mailboxImage = image
        let shouldSchedule = !mailboxPending
        if shouldSchedule { mailboxPending = true }
        mailboxLock.unlock()

        // Only schedule one MainActor task at a time; it will drain the mailbox.
        guard shouldSchedule else { return }

        Task { @MainActor in
            self.mailboxLock.lock()
            let f   = self.mailboxFrame
            let img = self.mailboxImage
            self.mailboxFrame   = nil
            self.mailboxImage   = nil
            self.mailboxPending = false
            self.mailboxLock.unlock()

            guard let f, let img else { return }
            self.latestFrame   = f
            self.procImage     = img
            self.trackingState = f.camera.trackingState
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in self.trackingState = .notAvailable }
    }

    nonisolated func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor in self.trackingState = .notAvailable }
    }
}

import AVFoundation
import UIKit
import Combine

class CameraManager: NSObject, ObservableObject {
    @Published var currentFrame: UIImage?
    /// Current video rotation angle applied to the output connection.
    /// CameraPreviewView observes this to keep the preview layer in sync.
    @Published private(set) var videoRotationAngle: CGFloat = 0

    let captureSession = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.daiki.camera", qos: .userInteractive)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private var captureDevice: AVCaptureDevice?

    override init() {
        super.init()
        setupCamera()
    }

    private func setupCamera() {
        captureSession.beginConfiguration()
        defer { captureSession.commitConfiguration() }

        captureSession.sessionPreset = .inputPriority

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              captureSession.canAddInput(input) else { return }
        captureSession.addInput(input)
        captureDevice = device

        // ── Pick best 60fps format at or near 1280×720 ──────────────────
        let target60 = pickFormat(device: device, preferredWidth: 1280, preferredHeight: 720, targetFPS: 60)
        let target30 = pickFormat(device: device, preferredWidth: 1280, preferredHeight: 720, targetFPS: 30)

        if let (fmt, fps) = target60 ?? target30 {
            do {
                try device.lockForConfiguration()
                device.activeFormat = fmt
                let dur = CMTime(value: 1, timescale: CMTimeScale(fps))
                device.activeVideoMinFrameDuration = dur
                device.activeVideoMaxFrameDuration = dur
                device.unlockForConfiguration()
                let dims = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
                print("[CameraManager] Format: \(dims.width)×\(dims.height) @ \(fps)fps")
            } catch {
                print("[CameraManager] lockForConfiguration error: \(error)")
            }
        }

        // ── Output ───────────────────────────────────────────────────────
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        guard captureSession.canAddOutput(videoOutput) else { return }
        captureSession.addOutput(videoOutput)

        // Apply initial rotation (landscape-right = 0°).
        applyRotationAngle(0)
    }

    /// Call from the main thread when the interface orientation changes.
    /// Updates both the output connection and publishes the angle for the preview layer.
    func updateInterfaceOrientation(_ orientation: UIInterfaceOrientation) {
        let angle: CGFloat
        switch orientation {
        case .portrait:             angle = 90
        case .portraitUpsideDown:   angle = 270
        case .landscapeLeft:        angle = 180
        case .landscapeRight:       angle = 0
        default:                    angle = 0
        }
        applyRotationAngle(angle)
    }

    private func applyRotationAngle(_ angle: CGFloat) {
        videoRotationAngle = angle
        guard let conn = videoOutput.connection(with: .video) else { return }
        if #available(iOS 17.0, *) {
            if conn.isVideoRotationAngleSupported(angle) {
                conn.videoRotationAngle = angle
            }
        }
    }

    /// Find a format that supports `targetFPS`, with dimensions closest to `preferredWidth × preferredHeight`.
    private func pickFormat(
        device: AVCaptureDevice,
        preferredWidth: Int32,
        preferredHeight: Int32,
        targetFPS: Double
    ) -> (AVCaptureDevice.Format, Double)? {
        var best: (AVCaptureDevice.Format, Double, Int64)?

        for fmt in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(fmt.formatDescription)
            guard let range = fmt.videoSupportedFrameRateRanges.first(where: { $0.maxFrameRate >= targetFPS })
            else { continue }

            let diff = abs(Int64(dims.width) - Int64(preferredWidth)) + abs(Int64(dims.height) - Int64(preferredHeight))
            if best == nil || diff < best!.2 {
                best = (fmt, min(range.maxFrameRate, targetFPS), diff)
            }
        }
        return best.map { ($0.0, $0.1) }
    }

    func startCapture() {
        queue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    func stopCapture() {
        captureSession.stopRunning()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // videoRotationAngle on the connection already applies the correct rotation,
        // so no manual CIImage orientation correction is needed here.
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let image = UIImage(cgImage: cgImage)

        DispatchQueue.main.async { [weak self] in
            self?.currentFrame = image
        }
    }
}

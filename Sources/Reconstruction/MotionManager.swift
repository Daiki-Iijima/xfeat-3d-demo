import CoreMotion
import simd

/// Wraps CMMotionManager to provide attitude (rotation) data between frames.
final class MotionManager: @unchecked Sendable {
    static let shared = MotionManager()

    private let motionManager = CMMotionManager()
    private var latestMotion: CMDeviceMotion?

    private init() {}

    var isAvailable: Bool { motionManager.isDeviceMotionAvailable }

    func start() {
        guard isAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical,
                                                to: .main) { [weak self] motion, _ in
            self?.latestMotion = motion
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
    }

    /// Current rotation matrix (world→device) from CoreMotion.
    /// Returns identity if unavailable.
    var rotationMatrix: simd_float3x3 {
        guard let r = latestMotion?.attitude.rotationMatrix else {
            return matrix_identity_float3x3
        }
        return simd_float3x3(
            SIMD3<Float>(Float(r.m11), Float(r.m21), Float(r.m31)),
            SIMD3<Float>(Float(r.m12), Float(r.m22), Float(r.m32)),
            SIMD3<Float>(Float(r.m13), Float(r.m23), Float(r.m33))
        )
    }

    /// Current gravity vector in device frame.
    var gravity: SIMD3<Float> {
        guard let g = latestMotion?.gravity else { return SIMD3<Float>(0, -1, 0) }
        return SIMD3<Float>(Float(g.x), Float(g.y), Float(g.z))
    }

    /// Capture the current attitude for later delta computation.
    func snapshot() -> simd_float3x3 { rotationMatrix }

    /// Rotation from `prev` attitude to current attitude (R_delta = R_cur * R_prev^T).
    func deltaSince(_ prev: simd_float3x3) -> simd_float3x3 {
        return rotationMatrix * prev.transpose
    }
}

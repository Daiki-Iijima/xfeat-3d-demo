import UIKit
import Accelerate

extension UIImage {

    /// Renders the image into a grayscale float array of size `width × height`, values in [0, 1].
    func toGrayscaleFloat(width: Int, height: Int) -> [Float]? {
        let targetSize = CGSize(width: width, height: height)

        // Draw into a grayscale 8-bit context
        let colorSpace = CGColorSpaceCreateDeviceGray()
        var pixels = [UInt8](repeating: 0, count: width * height)
        guard let ctx = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }

        ctx.draw(scaled(to: targetSize).cgImage ?? cgImage!, in: CGRect(origin: .zero, size: targetSize))

        // UInt8 → Float / 255
        var floats = [Float](repeating: 0, count: width * height)
        var scale: Float = 1.0 / 255.0
        vDSP_vfltu8(pixels, 1, &floats, 1, vDSP_Length(width * height))
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(width * height))
        return floats
    }

    /// Returns a planar RGB float array of size `3 × width × height`, values in [0, 1].
    ///
    /// Layout: all R pixels, then all G pixels, then all B pixels.
    /// ImageNet normalisation is intentionally NOT applied here; it is baked
    /// into the ALIKED CoreML model via registered buffers.
    func toRGBFloat(width: Int, height: Int) -> [Float]? {
        let targetSize = CGSize(width: width, height: height)
        let n = width * height

        var pixels = [UInt8](repeating: 0, count: n * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.draw(
            scaled(to: targetSize).cgImage ?? cgImage!,
            in: CGRect(origin: .zero, size: targetSize)
        )

        // Deinterleave RGBA → planar R / G / B float [0, 1]
        // Use vDSP_vfltu8 with stride-4 to skip Alpha channel efficiently.
        var rPlane = [Float](repeating: 0, count: n)
        var gPlane = [Float](repeating: 0, count: n)
        var bPlane = [Float](repeating: 0, count: n)

        pixels.withUnsafeBufferPointer { buf in
            let p = buf.baseAddress!
            vDSP_vfltu8(p.advanced(by: 0), 4, &rPlane, 1, vDSP_Length(n))
            vDSP_vfltu8(p.advanced(by: 1), 4, &gPlane, 1, vDSP_Length(n))
            vDSP_vfltu8(p.advanced(by: 2), 4, &bPlane, 1, vDSP_Length(n))
        }

        var result = [Float](repeating: 0, count: n * 3)
        var scale: Float = 1.0 / 255.0
        vDSP_vsmul(rPlane, 1, &scale, &result,           1, vDSP_Length(n))
        vDSP_vsmul(gPlane, 1, &scale, &result[n],        1, vDSP_Length(n))
        vDSP_vsmul(bPlane, 1, &scale, &result[n * 2],    1, vDSP_Length(n))
        return result
    }

    /// Returns the image scaled to `size`, always at scale 1.0 (pixel-exact).
    func scaled(to size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}

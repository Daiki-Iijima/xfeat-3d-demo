#import "OpenCVBridge.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#import <opencv2/opencv.hpp>
#import <opencv2/features2d.hpp>
#import <opencv2/calib3d.hpp>
#import <opencv2/aruco.hpp>
#pragma clang diagnostic pop

// ---------------------------------------------------------------------------
// Private helpers (accessible to all categories via class extension)
// ---------------------------------------------------------------------------

@interface OpenCVBridge ()
+ (cv::Mat)matFromUIImage:(UIImage *)image;
@end

@implementation OpenCVBridge

+ (cv::Mat)matFromUIImage:(UIImage *)image {
    CGImageRef cgImage = image.CGImage;
    size_t width  = CGImageGetWidth(cgImage);
    size_t height = CGImageGetHeight(cgImage);
    cv::Mat mat((int)height, (int)width, CV_8UC4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        mat.data, width, height, 8, mat.step[0],
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault
    );
    CGContextDrawImage(ctx, CGRectMake(0, 0, width, height), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_RGBA2BGR);
    return bgr;
}

// ── LK Optical Flow ────────────────────────────────────────────────────────

+ (NSData *)toGray:(UIImage *)image width:(int)width height:(int)height {
    cv::Mat bgr = [[self class] matFromUIImage:image];
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    return [NSData dataWithBytes:gray.data length:(size_t)(width * height)];
}

+ (LKTrackingResult *)trackLK:(NSData *)prevPointsXY
                         count:(NSInteger)count
                      prevGray:(NSData *)prevGray
                      currGray:(NSData *)currGray
                         width:(int)width
                        height:(int)height {
    if (count <= 0) {
        LKTrackingResult *empty = [[LKTrackingResult alloc] init];
        empty.count = 0;
        empty.pointsXY = [NSData data];
        empty.status   = [NSData data];
        return empty;
    }

    cv::Mat prevMat(height, width, CV_8UC1, (void *)prevGray.bytes);
    cv::Mat currMat(height, width, CV_8UC1, (void *)currGray.bytes);

    const float *src = (const float *)prevPointsXY.bytes;
    std::vector<cv::Point2f> prevPts((size_t)count), nextPts;
    for (NSInteger i = 0; i < count; i++) {
        prevPts[(size_t)i] = cv::Point2f(src[i * 2], src[i * 2 + 1]);
    }

    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        prevMat, currMat, prevPts, nextPts, status, err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
        0, 1e-4
    );

    std::vector<float> outPts((size_t)(count * 2));
    for (NSInteger i = 0; i < count; i++) {
        outPts[(size_t)(i * 2)]     = nextPts[(size_t)i].x;
        outPts[(size_t)(i * 2 + 1)] = nextPts[(size_t)i].y;
    }

    LKTrackingResult *result = [[LKTrackingResult alloc] init];
    result.count    = count;
    result.pointsXY = [NSData dataWithBytes:outPts.data() length:(size_t)(count * 2) * sizeof(float)];
    result.status   = [NSData dataWithBytes:status.data() length:(size_t)count];
    return result;
}

@end  // OpenCVBridge

// ---------------------------------------------------------------------------
// LKTrackingResult
// ---------------------------------------------------------------------------

@implementation LKTrackingResult
@end

// ---------------------------------------------------------------------------
// ORB Keypoint Detection
// ---------------------------------------------------------------------------

@interface ORBKeypoints ()
@property (nonatomic, readwrite) NSInteger count;
@property (nonatomic, strong, readwrite) NSData *keypointsXY;
@end

@implementation ORBKeypoints
@end

@implementation OpenCVBridge (ORBExtraction)

+ (nullable ORBKeypoints *)detectORBFrom:(UIImage *)image
                                    topK:(int)topK
                               procWidth:(int)procWidth
                              procHeight:(int)procHeight {
    cv::Mat bgr = [[self class] matFromUIImage:image];
    if (bgr.empty()) return nil;
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(procWidth, procHeight), 0, 0, cv::INTER_LINEAR);
    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);

    auto detector = cv::ORB::create(topK, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    std::vector<cv::KeyPoint> kps;
    detector->detect(gray, kps);
    std::sort(kps.begin(), kps.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response > b.response;
    });
    if ((int)kps.size() > topK) kps.resize((size_t)topK);

    int N = (int)kps.size();
    if (N == 0) return nil;

    std::vector<float> buf((size_t)(N * 2));
    for (int i = 0; i < N; ++i) {
        buf[(size_t)(i * 2)]     = kps[(size_t)i].pt.x;
        buf[(size_t)(i * 2 + 1)] = kps[(size_t)i].pt.y;
    }

    ORBKeypoints *result = [[ORBKeypoints alloc] init];
    result.count       = N;
    result.keypointsXY = [NSData dataWithBytes:buf.data() length:(size_t)N * 2 * sizeof(float)];
    return result;
}

@end  // OpenCVBridge (ORBExtraction)

// ---------------------------------------------------------------------------
// Pose Recovery + Triangulation
// ---------------------------------------------------------------------------

@interface PoseResult ()
@property (nonatomic, readwrite) BOOL success;
@property (nonatomic, strong, readwrite) NSData *rotationMatrix;
@property (nonatomic, strong, readwrite) NSData *translationVector;
@property (nonatomic, readwrite) NSInteger inlierCount;
@end

@implementation PoseResult
@end

@implementation OpenCVBridge (Reconstruction)

+ (PoseResult *)recoverPoseFrom:(NSData *)pts1XY
                           pts2:(NSData *)pts2XY
                          count:(NSInteger)count
                             fx:(float)fx fy:(float)fy
                             cx:(float)cx cy:(float)cy {
    PoseResult *fail = [[PoseResult alloc] init];
    fail.success = NO;
    fail.rotationMatrix    = [NSData data];
    fail.translationVector = [NSData data];
    fail.inlierCount = 0;

    if (count < 8) return fail;

    const float *p1 = (const float *)pts1XY.bytes;
    const float *p2 = (const float *)pts2XY.bytes;

    std::vector<cv::Point2f> prev((size_t)count), curr((size_t)count);
    for (NSInteger i = 0; i < count; i++) {
        prev[(size_t)i] = cv::Point2f(p1[i * 2], p1[i * 2 + 1]);
        curr[(size_t)i] = cv::Point2f(p2[i * 2], p2[i * 2 + 1]);
    }

    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    cv::Mat inlierMask;
    cv::Mat E = cv::findEssentialMat(prev, curr, K, cv::RANSAC, 0.999, 1.0, inlierMask);
    if (E.empty()) return fail;

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, prev, curr, K, R, t, inlierMask);
    if (inliers < 8) return fail;

    std::vector<float> rBuf(9);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            rBuf[(size_t)(r * 3 + c)] = (float)R.at<double>(r, c);

    std::vector<float> tBuf(3);
    for (int i = 0; i < 3; i++)
        tBuf[(size_t)i] = (float)t.at<double>(i);

    PoseResult *result = [[PoseResult alloc] init];
    result.success           = YES;
    result.rotationMatrix    = [NSData dataWithBytes:rBuf.data() length:9 * sizeof(float)];
    result.translationVector = [NSData dataWithBytes:tBuf.data() length:3 * sizeof(float)];
    result.inlierCount       = (NSInteger)inliers;
    return result;
}

+ (nullable NSData *)triangulatePoints:(NSData *)proj1
                                  proj2:(NSData *)proj2
                                  pts1:(NSData *)pts1XY
                                  pts2:(NSData *)pts2XY
                                 count:(NSInteger)count {
    if (count < 4) return nil;

    const float *p1f = (const float *)proj1.bytes;
    const float *p2f = (const float *)proj2.bytes;

    cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
    for (int i = 0; i < 12; i++) {
        P1.at<double>(i / 4, i % 4) = (double)p1f[i];
        P2.at<double>(i / 4, i % 4) = (double)p2f[i];
    }

    const float *pp1 = (const float *)pts1XY.bytes;
    const float *pp2 = (const float *)pts2XY.bytes;

    cv::Mat pts1Mat(2, (int)count, CV_64F);
    cv::Mat pts2Mat(2, (int)count, CV_64F);
    for (NSInteger i = 0; i < count; i++) {
        pts1Mat.at<double>(0, (int)i) = (double)pp1[i * 2];
        pts1Mat.at<double>(1, (int)i) = (double)pp1[i * 2 + 1];
        pts2Mat.at<double>(0, (int)i) = (double)pp2[i * 2];
        pts2Mat.at<double>(1, (int)i) = (double)pp2[i * 2 + 1];
    }

    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, pts1Mat, pts2Mat, pts4D);

    // Camera 2 centre in camera-1 frame: C2 = -R2^T * t2
    cv::Mat R2 = P2(cv::Rect(0, 0, 3, 3));
    cv::Mat t2 = P2(cv::Rect(3, 0, 1, 3));
    cv::Mat C2 = -R2.t() * t2;

    const double minCosParallax = std::cos(1.0 * CV_PI / 180.0);
    std::vector<float> out((size_t)(count * 3), 0.0f);
    for (NSInteger i = 0; i < count; i++) {
        double w = pts4D.at<double>(3, (int)i);
        if (std::abs(w) < 1e-8) continue;

        double X = pts4D.at<double>(0, (int)i) / w;
        double Y = pts4D.at<double>(1, (int)i) / w;
        double Z = pts4D.at<double>(2, (int)i) / w;

        // Cheirality: in front of camera 1
        if (Z <= 0) continue;

        // Cheirality: in front of camera 2
        cv::Mat pt3 = (cv::Mat_<double>(3,1) << X, Y, Z);
        cv::Mat pt_cam2 = R2 * pt3 + t2;
        if (pt_cam2.at<double>(2) <= 0) continue;

        // Parallax angle > 1°
        cv::Mat ray1 = pt3 / cv::norm(pt3);
        cv::Mat ray2 = (pt3 - C2) / cv::norm(pt3 - C2);
        if (ray1.dot(ray2) > minCosParallax) continue;

        out[(size_t)(i * 3)]     = (float)X;
        out[(size_t)(i * 3 + 1)] = (float)Y;
        out[(size_t)(i * 3 + 2)] = (float)Z;
    }

    return [NSData dataWithBytes:out.data() length:(size_t)(count * 3) * sizeof(float)];
}

@end  // OpenCVBridge (Reconstruction)

// ---------------------------------------------------------------------------
// ArUco 5×5 Marker Detection
// ---------------------------------------------------------------------------

@interface ArucoResult ()
@property (nonatomic, readwrite) BOOL detected;
@property (nonatomic, readwrite) NSInteger markerId;
@property (nonatomic, strong, readwrite) NSData *corners;
@property (nonatomic, strong, readwrite) NSData *rotationMatrix;
@property (nonatomic, strong, readwrite) NSData *translationVector;
@end

@implementation ArucoResult
@end

@implementation OpenCVBridge (ArUco)

+ (ArucoResult *)detectArUco5x5:(UIImage *)image
                     markerSize:(float)markerSizeMeters
                      procWidth:(int)procWidth
                     procHeight:(int)procHeight
                             fx:(float)fx fy:(float)fy
                             cx:(float)cx cy:(float)cy {
    ArucoResult *fail = [[ArucoResult alloc] init];
    fail.detected           = NO;
    fail.markerId           = -1;
    fail.corners            = [NSData data];
    fail.rotationMatrix     = [NSData data];
    fail.translationVector  = [NSData data];

    cv::Mat bgr = [[self class] matFromUIImage:image];
    if (bgr.empty()) return fail;

    cv::Mat proc;
    cv::resize(bgr, proc, cv::Size(procWidth, procHeight), 0, 0, cv::INTER_LINEAR);
    cv::Mat gray;
    cv::cvtColor(proc, gray, cv::COLOR_BGR2GRAY);

    // makePtr heap-allocates the Dictionary so Ptr<Dictionary> can safely manage its lifetime.
    // Passing a stack address (&dict) where Ptr<Dictionary>& is expected causes heap corruption.
    cv::Ptr<cv::aruco::Dictionary> dict = cv::makePtr<cv::aruco::Dictionary>(
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250)
    );
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(gray, dict, markerCorners, markerIds);

    if (markerIds.empty()) return fail;

    const auto& corners = markerCorners[0];

    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(markerCorners, markerSizeMeters, K, distCoeffs, rvecs, tvecs);

    cv::Mat R_mat;
    cv::Rodrigues(rvecs[0], R_mat);

    std::vector<float> cornerBuf(8);
    for (int i = 0; i < 4; i++) {
        cornerBuf[(size_t)(i * 2)]     = corners[(size_t)i].x;
        cornerBuf[(size_t)(i * 2 + 1)] = corners[(size_t)i].y;
    }

    std::vector<float> rBuf(9);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            rBuf[(size_t)(r * 3 + c)] = (float)R_mat.at<double>(r, c);

    std::vector<float> tBuf(3);
    tBuf[0] = (float)tvecs[0][0];
    tBuf[1] = (float)tvecs[0][1];
    tBuf[2] = (float)tvecs[0][2];

    ArucoResult *result = [[ArucoResult alloc] init];
    result.detected           = YES;
    result.markerId           = (NSInteger)markerIds[0];
    result.corners            = [NSData dataWithBytes:cornerBuf.data() length:8 * sizeof(float)];
    result.rotationMatrix     = [NSData dataWithBytes:rBuf.data()     length:9 * sizeof(float)];
    result.translationVector  = [NSData dataWithBytes:tBuf.data()     length:3 * sizeof(float)];
    return result;
}

@end  // OpenCVBridge (ArUco)

// ---------------------------------------------------------------------------
// PnP Pose Estimation
// ---------------------------------------------------------------------------

@interface PnPResult ()
@property (nonatomic, readwrite) BOOL success;
@property (nonatomic, strong, readwrite) NSData *rotationMatrix;
@property (nonatomic, strong, readwrite) NSData *translationVector;
@property (nonatomic, readwrite) NSInteger inlierCount;
@end

@implementation PnPResult
@end

@implementation OpenCVBridge (PnP)

+ (PnPResult *)solvePnPFrom:(NSData *)points3D
                   points2D:(NSData *)points2D
                      count:(NSInteger)count
                         fx:(float)fx fy:(float)fy
                         cx:(float)cx cy:(float)cy
                 iterations:(NSInteger)iterations {
    PnPResult *fail = [[PnPResult alloc] init];
    fail.success = NO;
    fail.rotationMatrix    = [NSData data];
    fail.translationVector = [NSData data];
    fail.inlierCount = 0;

    if (count < 6) return fail;

    const float *p3 = (const float *)points3D.bytes;
    const float *p2 = (const float *)points2D.bytes;

    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    obj.reserve((size_t)count);
    img.reserve((size_t)count);
    for (NSInteger i = 0; i < count; i++) {
        obj.push_back(cv::Point3f(p3[i*3], p3[i*3+1], p3[i*3+2]));
        img.push_back(cv::Point2f(p2[i*2], p2[i*2+1]));
    }

    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec, tvec, inliersMat;
    bool ok = cv::solvePnPRansac(
        obj, img, K, distCoeffs,
        rvec, tvec,
        false,           // useExtrinsicGuess
        (int)iterations, // iterationsCount (caller-controlled: 60 tracking / 150 recovery)
        2.5f,            // reprojectionError (pixels) — tightened from 4.0
        0.99,            // confidence
        inliersMat,
        cv::SOLVEPNP_ITERATIVE
    );
    if (!ok) return fail;

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);

    std::vector<float> rBuf(9);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            rBuf[(size_t)(r * 3 + c)] = (float)R_mat.at<double>(r, c);

    std::vector<float> tBuf(3);
    tBuf[0] = (float)tvec.at<double>(0);
    tBuf[1] = (float)tvec.at<double>(1);
    tBuf[2] = (float)tvec.at<double>(2);

    PnPResult *result = [[PnPResult alloc] init];
    result.success = YES;
    result.rotationMatrix    = [NSData dataWithBytes:rBuf.data() length:9 * sizeof(float)];
    result.translationVector = [NSData dataWithBytes:tBuf.data() length:3 * sizeof(float)];
    result.inlierCount       = inliersMat.rows;
    return result;
}

@end  // OpenCVBridge (PnP)

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

// ---------------------------------------------------------------------------
// LK Optical Flow Tracking
// ---------------------------------------------------------------------------

@interface LKTrackingResult : NSObject
@property (nonatomic, strong) NSData *pointsXY;
@property (nonatomic, strong) NSData *status;
@property (nonatomic) NSInteger count;
@end

@interface OpenCVBridge : NSObject

/// Resize image to (width × height) and return packed uint8 grayscale bytes.
+ (NSData *)toGray:(UIImage *)image width:(int)width height:(int)height
    NS_SWIFT_NAME(toGray(_:width:height:));

/// Sparse Lucas-Kanade optical flow.
+ (LKTrackingResult *)trackLK:(NSData *)prevPointsXY
                         count:(NSInteger)count
                      prevGray:(NSData *)prevGray
                      currGray:(NSData *)currGray
                         width:(int)width
                        height:(int)height
    NS_SWIFT_NAME(trackLK(_:count:prevGray:currGray:width:height:));

@end

// ---------------------------------------------------------------------------
// ORB Keypoint Detection
// ---------------------------------------------------------------------------

@interface ORBKeypoints : NSObject
@property (nonatomic, readonly) NSInteger count;
@property (nonatomic, strong, readonly) NSData *keypointsXY;
@end

@interface OpenCVBridge (ORBExtraction)
+ (nullable ORBKeypoints *)detectORBFrom:(UIImage *)image
                                    topK:(int)topK
                               procWidth:(int)procWidth
                              procHeight:(int)procHeight
    NS_SWIFT_NAME(detectORB(from:topK:procWidth:procHeight:));
@end

// ---------------------------------------------------------------------------
// 3D Reconstruction: Essential Matrix + Triangulation
// ---------------------------------------------------------------------------

/// Result of Essential Matrix estimation and pose recovery.
@interface PoseResult : NSObject
/// Whether pose was successfully recovered.
@property (nonatomic, readonly) BOOL success;
/// 9 float32 values: row-major 3×3 rotation matrix R.
@property (nonatomic, strong, readonly) NSData *rotationMatrix;
/// 3 float32 values: translation vector t (unit length, up to scale).
@property (nonatomic, strong, readonly) NSData *translationVector;
/// Number of inlier correspondences used.
@property (nonatomic, readonly) NSInteger inlierCount;
@end

@interface OpenCVBridge (Reconstruction)

/// Recover relative camera pose (R, t) from 2D–2D point correspondences.
+ (PoseResult *)recoverPoseFrom:(NSData *)pts1XY
                           pts2:(NSData *)pts2XY
                          count:(NSInteger)count
                             fx:(float)fx fy:(float)fy
                             cx:(float)cx cy:(float)cy
    NS_SWIFT_NAME(recoverPose(pts1:pts2:count:fx:fy:cx:cy:));

/// Triangulate 3D points from two projection matrices and matched 2D points.
+ (nullable NSData *)triangulatePoints:(NSData *)proj1
                                  proj2:(NSData *)proj2
                                  pts1:(NSData *)pts1XY
                                  pts2:(NSData *)pts2XY
                                 count:(NSInteger)count
    NS_SWIFT_NAME(triangulatePoints(_:proj2:pts1:pts2:count:));

@end

// ---------------------------------------------------------------------------
// ArUco 5×5 Marker Detection
// ---------------------------------------------------------------------------

/// Result of ArUco marker detection and pose estimation.
@interface ArucoResult : NSObject
/// Whether a marker was detected.
@property (nonatomic, readonly) BOOL detected;
/// ID of the first detected marker (-1 if none).
@property (nonatomic, readonly) NSInteger markerId;
/// 8 float32 values: [x0,y0,x1,y1,x2,y2,x3,y3] corners in procWidth×procHeight image coords.
@property (nonatomic, strong, readonly) NSData *corners;
/// 9 float32 values: row-major 3×3 rotation matrix (marker → camera frame).
@property (nonatomic, strong, readonly) NSData *rotationMatrix;
/// 3 float32 values: translation vector in metres (marker centre in camera frame).
@property (nonatomic, strong, readonly) NSData *translationVector;
@end

@interface OpenCVBridge (ArUco)

/// Detect the first 5×5 ArUco marker in the image and estimate its pose.
/// Corners are returned in procWidth×procHeight coordinate space.
/// @param image            Source frame (any size; resized internally).
/// @param markerSizeMeters Physical side-length of the marker in metres.
/// @param procWidth        Width of the processing space (e.g. 960).
/// @param procHeight       Height of the processing space (e.g. 720).
/// @param fx               Focal length x in proc-space pixels.
/// @param fy               Focal length y.
/// @param cx               Principal point x.
/// @param cy               Principal point y.
+ (ArucoResult *)detectArUco5x5:(UIImage *)image
                     markerSize:(float)markerSizeMeters
                      procWidth:(int)procWidth
                     procHeight:(int)procHeight
                             fx:(float)fx fy:(float)fy
                             cx:(float)cx cy:(float)cy
    NS_SWIFT_NAME(detectArUco5x5(_:markerSize:procWidth:procHeight:fx:fy:cx:cy:));

@end

// ---------------------------------------------------------------------------
// PnP Pose Estimation
// ---------------------------------------------------------------------------

/// Result of PnP pose estimation.
@interface PnPResult : NSObject
/// Whether pose was successfully recovered.
@property (nonatomic, readonly) BOOL success;
/// 9 float32 values: row-major 3×3 rotation matrix R (world-to-camera).
@property (nonatomic, strong, readonly) NSData *rotationMatrix;
/// 3 float32 values: translation vector t (world-to-camera).
@property (nonatomic, strong, readonly) NSData *translationVector;
/// Number of inlier correspondences.
@property (nonatomic, readonly) NSInteger inlierCount;
@end

@interface OpenCVBridge (PnP)

/// Estimate camera pose from 3D–2D point correspondences using solvePnPRansac.
/// @param points3D  Flat float32 array: [x0,y0,z0, x1,y1,z1, ...] (3×count floats).
/// @param points2D  Flat float32 array: [u0,v0, u1,v1, ...] (2×count floats, proc-space).
/// @param count     Number of correspondences.
/// @param fx,fy,cx,cy  Camera intrinsics at proc resolution.
+ (PnPResult *)solvePnPFrom:(NSData *)points3D
                   points2D:(NSData *)points2D
                      count:(NSInteger)count
                         fx:(float)fx fy:(float)fy
                         cx:(float)cx cy:(float)cy
                 iterations:(NSInteger)iterations
    NS_SWIFT_NAME(solvePnP(points3D:points2D:count:fx:fy:cx:cy:iterations:));

@end

NS_ASSUME_NONNULL_END

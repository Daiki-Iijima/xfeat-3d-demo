#pragma once
#import <Foundation/Foundation.h>
#import <simd/simd.h>

/// A single 2D observation: which 3-D point was seen from which pose frame,
/// and at what image coordinate.
typedef struct BAObservation {
    int   poseIdx;   ///< Index into the poses array (0 = reference frame).
    int   pointIdx;  ///< Index into the points array.
    float u;         ///< Observed x pixel (proc resolution).
    float v;         ///< Observed y pixel (proc resolution).
} BAObservation;

/// Result returned by the adjuster.
typedef struct BAResult {
    /// Refined camera-to-world poses (same count and order as input).
    simd_float4x4 *poses;     ///< Caller-owned; free with free().
    int            poseCount;

    /// Refined 3-D point positions (same count and order as input).
    simd_float3   *points;    ///< Caller-owned; free with free().
    int            pointCount;

    double         finalCost;
    bool           converged;
} BAResult;

#ifdef __cplusplus
extern "C" {
#endif

/// Run a sparse Local Bundle Adjustment using Ceres Solver.
///
/// @param poses         Camera-to-world 4×4 matrices (column-major simd layout).
/// @param poseCount     Number of poses.
/// @param points        3-D landmark positions in world space.
/// @param pointCount    Number of landmarks.
/// @param observations  2-D reprojection observations.
/// @param obsCount      Number of observations.
/// @param fx            Focal length x (proc resolution).
/// @param fy            Focal length y.
/// @param cx            Principal point x.
/// @param cy            Principal point y.
/// @param fixFirstPose  If true, the first pose is held fixed (gauge freedom fix).
/// @return              Refined poses + points.  Caller must free result.poses and result.points.
BAResult runLocalBA(const simd_float4x4 *poses,
                    int                  poseCount,
                    const simd_float3   *points,
                    int                  pointCount,
                    const BAObservation *observations,
                    int                  obsCount,
                    float fx, float fy, float cx, float cy,
                    bool fixFirstPose);

#ifdef __cplusplus
}
#endif

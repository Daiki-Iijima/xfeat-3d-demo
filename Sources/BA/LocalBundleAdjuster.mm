// LocalBundleAdjuster.mm
// Sparse Local Bundle Adjustment via Ceres Solver.
//
// Pose parameterisation: angle-axis (3) + translation (3) — 6 DOF.
// Cost: reprojection error (Huber loss, delta = 1.0 px).

#include "LocalBundleAdjuster.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstring>

// ---------------------------------------------------------------------------
// Reprojection cost functor
// ---------------------------------------------------------------------------
// Residual: r = projected(pose, point) - observed   (2-D)
// pose  : [angle_axis(3), translation(3)]  — world-to-camera (R,t)
// point : [x, y, z]                        — world coordinates
struct ReprojectionError {
    ReprojectionError(double u, double v,
                      double fx, double fy,
                      double cx, double cy)
        : obs_u(u), obs_v(v), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const pose,   // [aa0,aa1,aa2, tx,ty,tz]
                    const T* const point,  // [X, Y, Z]
                    T* residuals) const {
        // Rotate: p_cam = R * point + t
        T p[3];
        ceres::AngleAxisRotatePoint(pose, point, p);
        p[0] += pose[3];
        p[1] += pose[4];
        p[2] += pose[5];

        // Project (pinhole)
        T inv_z = T(1.0) / p[2];
        T u_proj = T(fx_) * p[0] * inv_z + T(cx_);
        T v_proj = T(fy_) * p[1] * inv_z + T(cy_);

        residuals[0] = u_proj - T(obs_u);
        residuals[1] = v_proj - T(obs_v);
        return true;
    }

    static ceres::CostFunction* Create(double u, double v,
                                       double fx, double fy,
                                       double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(u, v, fx, fy, cx, cy));
    }

    double obs_u, obs_v;
    double fx_, fy_, cx_, cy_;
};

// ---------------------------------------------------------------------------
// Helpers: convert between simd_float4x4 and angle-axis+translation
// ---------------------------------------------------------------------------
static void poseToParams(const simd_float4x4& cam2world, double* params) {
    // cam2world → world-to-cam = inverse
    // For a rigid transform: R_wc = cam2world.R^T, t_wc = -R_wc * cam2world.t
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            R(r, c) = cam2world.columns[c][r];  // simd is column-major

    Eigen::Vector3d t_cw(
        cam2world.columns[3][0],
        cam2world.columns[3][1],
        cam2world.columns[3][2]);

    // world-to-cam rotation and translation
    Eigen::Matrix3d Rwc = R.transpose();
    Eigen::Vector3d twc = -Rwc * t_cw;

    // Rotation → angle-axis
    Eigen::AngleAxisd aa(Rwc);
    Eigen::Vector3d aaVec = aa.angle() * aa.axis();

    params[0] = aaVec[0];
    params[1] = aaVec[1];
    params[2] = aaVec[2];
    params[3] = twc[0];
    params[4] = twc[1];
    params[5] = twc[2];
}

static simd_float4x4 paramsToC2W(const double* params) {
    // params: [aa(3), t_wc(3)]
    Eigen::Vector3d aaVec(params[0], params[1], params[2]);
    double angle = aaVec.norm();
    Eigen::AngleAxisd aa(angle < 1e-9 ? 0.0 : angle,
                         angle < 1e-9 ? Eigen::Vector3d::UnitX() : aaVec.normalized());
    Eigen::Matrix3d Rwc = aa.toRotationMatrix();
    Eigen::Vector3d twc(params[3], params[4], params[5]);

    // cam2world: R = Rwc^T, t = -R * twc
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;

    simd_float4x4 m = matrix_identity_float4x4;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            m.columns[c][r] = (float)Rcw(r, c);
    m.columns[3][0] = (float)tcw[0];
    m.columns[3][1] = (float)tcw[1];
    m.columns[3][2] = (float)tcw[2];
    return m;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
BAResult runLocalBA(const simd_float4x4 *poses,
                    int                  poseCount,
                    const simd_float3   *points,
                    int                  pointCount,
                    const BAObservation *observations,
                    int                  obsCount,
                    float fx, float fy, float cx, float cy,
                    bool fixFirstPose) {

    // --- Allocate parameter blocks ---
    // poses: poseCount × 6  (angle-axis + translation, world-to-cam)
    // points: pointCount × 3
    std::vector<std::array<double,6>> poseParams(poseCount);
    std::vector<std::array<double,3>> pointParams(pointCount);

    for (int i = 0; i < poseCount; i++)
        poseToParams(poses[i], poseParams[i].data());

    for (int j = 0; j < pointCount; j++) {
        pointParams[j][0] = points[j].x;
        pointParams[j][1] = points[j].y;
        pointParams[j][2] = points[j].z;
    }

    // --- Build Ceres problem ---
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(1.0);  // 1 px Huber

    for (int k = 0; k < obsCount; k++) {
        const BAObservation& obs = observations[k];
        if (obs.poseIdx < 0 || obs.poseIdx >= poseCount)   continue;
        if (obs.pointIdx < 0 || obs.pointIdx >= pointCount) continue;

        ceres::CostFunction* cost = ReprojectionError::Create(
            obs.u, obs.v, fx, fy, cx, cy);

        problem.AddResidualBlock(cost, loss,
                                 poseParams[obs.poseIdx].data(),
                                 pointParams[obs.pointIdx].data());
    }

    // Fix gauge freedom: hold the first pose constant
    if (fixFirstPose && poseCount > 0) {
        problem.SetParameterBlockConstant(poseParams[0].data());
    }

    // --- Solve ---
    ceres::Solver::Options opts;
    opts.linear_solver_type           = ceres::DENSE_SCHUR;
    opts.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
    opts.max_num_iterations           = 50;
    opts.num_threads                  = 2;
    opts.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // --- Pack result ---
    BAResult result;
    result.poseCount  = poseCount;
    result.pointCount = pointCount;
    result.finalCost  = summary.final_cost;
    result.converged  = (summary.termination_type == ceres::CONVERGENCE ||
                         summary.termination_type == ceres::USER_SUCCESS);

    result.poses  = (simd_float4x4*)malloc(poseCount  * sizeof(simd_float4x4));
    result.points = (simd_float3*)  malloc(pointCount * sizeof(simd_float3));

    for (int i = 0; i < poseCount; i++)
        result.poses[i] = paramsToC2W(poseParams[i].data());

    for (int j = 0; j < pointCount; j++) {
        result.points[j].x = (float)pointParams[j][0];
        result.points[j].y = (float)pointParams[j][1];
        result.points[j].z = (float)pointParams[j][2];
    }

    return result;
}

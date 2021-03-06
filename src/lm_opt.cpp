// Given body poses over time, compute camera extrinsic pose.
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"
#include "common_cpp/transform.h"

using namespace std;
using namespace Eigen;

typedef vector<Vector2d, aligned_allocator<Vector2d>> pix_vec;
typedef vector<pix_vec, aligned_allocator<pix_vec>> img_vec;


struct QuatPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    common::Quaternion<T> q(x);
    Map<const Matrix<T,3,1>> d(delta);
    Map<Matrix<T,4,1>> qp(x_plus_delta);
    qp = (q + d).toEigen();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<QuatPlus, 4, 3> QuatLocalParam;



struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    common::Transform<T> q(x);
    Map<const Matrix<T,6,1>> d(delta);
    Map<Matrix<T,7,1>> qp(x_plus_delta);
    qp = (q + d).toEigen();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformLocalParam;



class Feature
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Feature(const Vector2d& _pix1, const Vector2d& _pix2, const common::Transformd& _x1_ib, const common::Transformd& _x2_ib, const Matrix3d& cam_mat)
  {
    pix1 = _pix1;
    pix2 = _pix2;
    x1_ib = _x1_ib;
    x2_ib = _x2_ib;
    camera_matrix = cam_mat;
  }

  template<typename T>
  bool operator()(const T* const _x_bc, const T* const _rho1, T* residuals) const
  {
    // Copy/map states and residual error
    const common::Transform<T> x_bc(_x_bc);
    const T rho1(_rho1[0]);
    Map<Matrix<T,2,1>> r(residuals);

    // Convert measurements, etc. to templated types
    Matrix<T,3,3> K_ = camera_matrix.cast<T>();
    Matrix<T,2,1> pix1_ = pix1.cast<T>();
    Matrix<T,2,1> pix2_ = pix2.cast<T>();
    common::Transform<T> x1_ib_, x2_ib_;
    x1_ib_.p(x1_ib.p().cast<T>());
    x1_ib_.q(x1_ib.q().toEigen().cast<T>());
    x2_ib_.p(x2_ib.p().cast<T>());
    x2_ib_.q(x2_ib.q().toEigen().cast<T>());

    // Predict landmark vector at second frame using estimated states
    Matrix<T,3,1> zeta1;
    common::unitVectorFromPixelPosition(zeta1,pix1_,K_);
    Matrix<T,3,1> p2_cl_hat = x_bc.q().rotp(x2_ib_.q().rotp(x1_ib_.q().rota(x_bc.q().rota(T(1.0) / rho1 * zeta1) +
                              x_bc.p()) + x1_ib_.p() - x2_ib_.p()) - x_bc.p());

    // Compute residual error
    Matrix<T,2,1> pix2_hat;
    common::projectToImage(pix2_hat, p2_cl_hat, K_);
    r = pix2_ - pix2_hat;

    return true;
  }

  Matrix3d camera_matrix; // Camera intrinsic matrix
  Vector2d pix1, pix2; // pixels positions of measured landmark in frame 1 and 2, respectively
  common::Transformd x1_ib, x2_ib; // body pose at time of frame 1 and 2, respectively

};
typedef ceres::AutoDiffCostFunction<Feature, 2, 7, 1> FeatureFactor;



void bodyPose(const double& t, common::Transformd& x_ib)
{
  // Body pose as a function of time
  double x = sin(t);
  double y = sin(t);
  double z = cos(t);
  double phi = 0.1 * sin(t);
  double theta = 0.1 * cos(t);
  double psi = 0.1 * sin(t);

  // Replace toEigen
  x_ib.p(Vector3d(x, y, z));
  x_ib.q(common::Quaterniond::fromEuler(phi, theta, psi));
}


int main()
{
  // Constants
  srand((unsigned)time(NULL));

  // Build camera intrinsic matrix
  Matrix3d camera_matrix;
  camera_matrix << 483.4673,      0.0, 320.0,
                        0.0, 481.3655, 240.0,
                        0.0,      0.0,   1.0;

  // Landmark vectors in NED fixed frame
  Array<double, 3, 5> lm;
  lm.setRandom();
  lm.row(0) += 3;
  lm.row(1) *= 3;
  lm.row(2) *= 3;

  // Bady to camera translation and rotation
  Vector3d p_bc(0, 0, 0); // body to camera translation in body frame
  common::Quaterniond q_bcb = common::Quaterniond::fromEuler(0, 0, 0); // body to camera-body rotation
  common::Quaterniond q_cbc = common::Quaterniond::fromEuler(M_PI/2.0, 0.0, M_PI/2.0); // camera-body to camera rotation
  common::Quaterniond q_bc = q_bcb * q_cbc; // body to camera rotation
  common::Transformd x_bc(p_bc, q_bc);

  // Define times and poses of body
  double dt = 0.04;
  double tf = 10.0;
  double t = 0;
  vector<double> ts;
  vector<common::Transformd, aligned_allocator<common::Transformd>> x_ibs;
  common::Transformd x_ib;
  while (t <= tf)
  {
    bodyPose(t, x_ib);
    ts.push_back(t);
    x_ibs.push_back(x_ib);
    t += dt;
  }

  // Get pixel measurements of each position
  img_vec imgs;
  pix_vec pixs;
  Vector3d p_cl;
  Vector2d pix;
  for (int i = 0; i < x_ibs.size(); ++i)
  {
    common::Transformd x_ic;
    x_ic.p(x_ibs[i].p() + x_ibs[i].q().rota(x_bc.p()));
    x_ic.q(x_ibs[i].q() * x_bc.q());

    pixs.clear();
    for (int j = 0; j < lm.cols(); ++j)
    {
      p_cl = x_ic.q().rotp(lm.col(j).matrix() - x_ic.p()); // Get landmark vector in camera frame
      common::projectToImage(pix, p_cl, camera_matrix); // Project into camera image
      pixs.push_back(pix);
    }
    imgs.push_back(pixs);
  }

  // Compute true initial depths
  vector<double> rhos;
  for (int i = 0; i < lm.cols(); ++i)
    rhos.push_back(1.0 / (lm.col(i).matrix() - (x_ibs[0].p() + x_ibs[0].q().rota(x_bc.p()))).norm());

  bool test_error = false;
  if (test_error)
  {
    // Define position and attitude of two cameras in NED fixed frame
    common::Transformd x1_ib, x2_ib;
    double t1 = 0.0;
    double t2 = 1.0;
    bodyPose(t1, x1_ib);
    bodyPose(t2, x2_ib);

    common::Transformd x1_ic, x2_ic;
    x1_ic.p(x1_ib.p() + x1_ib.q().rota(x_bc.p()));
    x2_ic.p(x2_ib.p() + x2_ib.q().rota(x_bc.p()));
    x1_ic.q(x1_ib.q() * x_bc.q());
    x2_ic.q(x2_ib.q() * x_bc.q());

    // Get landmark vector in each camera frame
    Vector3d p1_cl = x1_ic.q().rotp(lm.col(0).matrix() - x1_ic.p());
    Vector3d p2_cl = x2_ic.q().rotp(lm.col(0).matrix() - x2_ic.p());

    // Project into camera images
    Vector2d nu1, nu2;
    common::projectToImage(nu1, p1_cl, camera_matrix);
    common::projectToImage(nu2, p2_cl, camera_matrix);

    // Project out of images
    Vector3d p1_cl_, p2_cl_;
    common::unitVectorFromPixelPosition(p1_cl_, nu1, camera_matrix);
    common::unitVectorFromPixelPosition(p2_cl_, nu2, camera_matrix);

    // Check the equation relating the pixels matched in each image
    Vector3d zeta1;
    Vector2d nu2_hat;
    double rho1 = 1.0 / p1_cl.norm();
    common::unitVectorFromPixelPosition(zeta1, nu1, camera_matrix);
    Vector3d p2_cl_hat = x_bc.q().rotp(x2_ib.q().rotp(x1_ib.q().rota(x_bc.q().rota(1.0 / rho1 * zeta1) +
                         x_bc.p()) + x1_ib.p() - x2_ib.p()) - x_bc.p());
    common::projectToImage(nu2_hat, p2_cl_hat, camera_matrix);

    cout << "nu1       =  " << nu1.transpose() << endl;
    cout << "nu2       = " << nu2.transpose() << endl;
    cout << "p1_cl     =  " << p1_cl.transpose() << endl;
    cout << "p2_cl     =  " << p2_cl.transpose() << endl;
    cout << "p2_cl_hat =  " << p2_cl_hat.transpose() << endl;
    cout << "distance =   " << p2_cl.norm() << endl;
    cout << "distance^2 = " << p2_cl.norm() * p2_cl.norm() << endl;
    cout << "product   =  " << p2_cl.dot(p2_cl_hat) << endl;
    cout << "nu2_hat        = " << nu2_hat.transpose() << endl;
    cout << "residual error = " << (nu2 - nu2_hat).transpose() << endl;
  }
  else
  {
    // Initialize estimated camera transform
    Vector3d noise;
    noise.setRandom();
    noise *= 0.1;
    common::Transformd x_bc_hat = x_bc;
    x_bc_hat.p(x_bc_hat.p() + noise);
    x_bc_hat.q(x_bc_hat.q() + noise);

    // Initialize landmark inverse distances based on body positions
    vector<double> rho_hats;
    for (int i = 0; i < lm.cols(); ++i)
      rho_hats.push_back(0.1);

    // Output initial comparisons with truth
    cout << "======= Initial Errors =======" << endl;
    cout << "x_bc error = " << (x_bc - x_bc_hat).transpose() << endl;
    for (int i = 0; i < rhos.size(); ++i)
      cout << "rho[" << i << "] (tru, est, err) = " << rhos[i] << ", " << rho_hats[i] << ", " << rhos[i] - rho_hats[i] << endl;

    // Build optimization problem with Ceres-Solver
    ceres::Problem problem;

    // Add parameters to solver
    problem.AddParameterBlock(x_bc_hat.data(), 7, new XformLocalParam);
    for (int i = 0; i < rho_hats.size(); ++i)
      problem.AddParameterBlock(&rho_hats[i], 1);

    // Add factors
    for (int i = 0; i < rho_hats.size(); ++i)
    {
      for (int j = 1; j < x_ibs.size(); ++j)
      {
        problem.AddResidualBlock(new FeatureFactor(
                                 new Feature(imgs[0][i], imgs[j][i], x_ibs[0], x_ibs[j], camera_matrix)),
                                 NULL, x_bc_hat.data(), &rho_hats[i]);
      }
    }

    // Solve for the optimal rotation and translation direciton
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.num_linear_solver_threads = 1;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "\n\n" << summary.BriefReport() << "\n\n";

    // Output final comparisons with truth
    cout << "\n======= Final Errors =======" << endl;
    cout << "x_bc error = " << (x_bc - x_bc_hat).transpose() << endl;
    for (int i = 0; i < rhos.size(); ++i)
      cout << "rho[" << i << "] (tru, est, err) = " << rhos[i] << ", " << rho_hats[i] << ", " << rhos[i] - rho_hats[i] << endl;
  }

  return 0;
}

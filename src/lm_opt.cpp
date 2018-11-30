// Given body poses over time, compute camera extrinsic pose.
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "geometry/quat.h"
#include "geometry/xform.h"
#include "geometry/support.h"
#include "geometry/cam.h"

using namespace std;
using namespace Eigen;

typedef vector<Vector2d, aligned_allocator<Vector2d>> pix_vec;
typedef vector<pix_vec, aligned_allocator<pix_vec>> img_vec;


struct QuatPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    Quat<T> q(x);
    Map<const Matrix<T,3,1>> d(delta);
    Map<Matrix<T,4,1>> qp(x_plus_delta);
    qp = (q + d).elements();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<QuatPlus, 4, 3> QuatLocalParam;



struct XformPlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const
  {
    xform::Xform<T> q(x);
    Map<const Matrix<T,6,1>> d(delta);
    Map<Matrix<T,7,1>> qp(x_plus_delta);
    qp = (q + d).elements();
    return true;
  }
};
typedef ceres::AutoDiffLocalParameterization<XformPlus, 7, 6> XformLocalParam;



class Feature
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Feature(const Vector2d& _pix1, const Vector2d& _pix2, const xform::Xformd& _x1_ib, const xform::Xformd& _x2_ib, const Camerad& _cam)
  {
    pix1 = _pix1;
    pix2 = _pix2;
    x1_ib = _x1_ib;
    x2_ib = _x2_ib;
    cam = _cam;
  }

  template<typename T>
  bool operator()(const T* const _x_bc, const T* const _rho1, T* residuals) const
  {
    // Copy/map states and residual error
    const xform::Xform<T> x_bc(_x_bc);
    const T rho1(_rho1[0]);
    Map<Matrix<T,2,1>> r(residuals);

    // Convert measurements, etc. to templated types
    Camera<T> cam_ = cam.cast<T>();
    Matrix<T,2,1> pix1_ = pix1.cast<T>();
    Matrix<T,2,1> pix2_ = pix2.cast<T>();
    xform::Xform<T> x1_ib_, x2_ib_;
    x1_ib_.t_ = x1_ib.t_.cast<T>();
    x1_ib_.q_.arr_ = x1_ib.q_.arr_.cast<T>();
    x2_ib_.t_ = x2_ib.t_.cast<T>();
    x2_ib_.q_.arr_ = x2_ib.q_.arr_.cast<T>();

    // Predict landmark vector at second frame using estimated states
    Matrix<T,3,1> zeta1;
    cam_.invProj(pix1_, T(1.0), zeta1);
    Matrix<T,3,1> p2_cl_hat = x_bc.q_.rotp(x2_ib_.q_.rotp(x1_ib_.q_.rota(x_bc.q_.rota(T(1.0) / rho1 * zeta1) +
                              x_bc.t_) + x1_ib_.t_ - x2_ib_.t_) - x_bc.t_);

    // Compute residual error
    Matrix<T,2,1> pix2_hat;
    cam_.proj(p2_cl_hat, pix2_hat);
    r = pix2_ - pix2_hat;

    return true;
  }

  Camerad cam;
  Vector2d pix1, pix2; // pixels positions of measured landmark in frame 1 and 2, respectively
  xform::Xformd x1_ib, x2_ib; // body pose at time of frame 1 and 2, respectively

};
typedef ceres::AutoDiffCostFunction<Feature, 2, 7, 1> FeatureFactor;



void bodyPose(const double& t, xform::Xformd& x_ib)
{
  // Body pose as a function of time
  double x = sin(t);
  double y = sin(t);
  double z = cos(t);
  double phi = 0.1 * sin(t);
  double theta = 0.1 * cos(t);
  double psi = 0.1 * sin(t);

  // Replace elements
  x_ib.sett(Vector3d(x, y, z));
  x_ib.setq(quat::Quatd(phi, theta, psi));
}


int main()
{
  // Constants
  srand((unsigned)time(NULL));

  // Build a camera
  Vector2d focal_len(483.4673, 481.3655);
  Vector2d cam_center(320.0, 240.0);
  Vector2d image_size(640.0, 480.0);
  Matrix<double, 5, 1> distortion;
  distortion << 0, 0, 0, 0, 0;
  double cam_skew(0);
  Camerad cam(focal_len, cam_center, distortion, cam_skew, image_size);

  // Landmark vectors in NED fixed frame
  Array<double, 3, 50> lm;
  lm.setRandom();
  lm.row(0) += 3;
  lm.row(1) *= 3;
  lm.row(2) *= 3;

  // Bady to camera translation and rotation
  Vector3d p_bc(0, 0, 0); // body to camera translation in body frame
  quat::Quatd q_bcb(0, 0, 0); // body to camera-body rotation
  quat::Quatd q_cbc(M_PI/2.0, 0.0, M_PI/2.0); // camera-body to camera rotation
  quat::Quatd q_bc = q_bcb * q_cbc; // body to camera rotation
  xform::Xformd x_bc(p_bc, q_bc);

  // Define times and poses of body
  double dt = 0.1;
  double tf = 1.0;
  double t = 0;
  vector<double> ts;
  vector<xform::Xformd, aligned_allocator<xform::Xformd>> x_ibs;
  xform::Xformd x_ib;
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
    xform::Xformd x_ic;
    x_ic.sett(x_ibs[i].t() + x_ibs[i].q().rota(x_bc.t()));
    x_ic.setq(x_ibs[i].q() * x_bc.q());

    pixs.clear();
    for (int j = 0; j < lm.cols(); ++j)
    {
      p_cl = x_ic.q().rotp(lm.col(j).matrix() - x_ic.t()); // Get landmark vector in camera frame
      cam.proj(p_cl, pix); // Project into camera image
      pixs.push_back(pix);
    }
    imgs.push_back(pixs);
  }

  // Compute true initial depths
  vector<double> rhos;
  for (int i = 0; i < lm.cols(); ++i)
    rhos.push_back(1.0 / (lm.col(i).matrix() - (x_ibs[0].t() + x_ibs[0].q().rota(x_bc.t()))).norm());

  bool test_error = false;
  if (test_error)
  {
    // Define position and attitude of two cameras in NED fixed frame
    xform::Xformd x1_ib, x2_ib;
    double t1 = 0.0;
    double t2 = 1.0;
    bodyPose(t1, x1_ib);
    bodyPose(t2, x2_ib);

    xform::Xformd x1_ic, x2_ic;
    x1_ic.sett(x1_ib.t() + x1_ib.q().rota(x_bc.t()));
    x2_ic.sett(x2_ib.t() + x2_ib.q().rota(x_bc.t()));
    x1_ic.setq(x1_ib.q() * x_bc.q());
    x2_ic.setq(x2_ib.q() * x_bc.q());

    // Get landmark vector in each camera frame
    Vector3d p1_cl = x1_ic.q().rotp(lm.col(0).matrix() - x1_ic.t());
    Vector3d p2_cl = x2_ic.q().rotp(lm.col(0).matrix() - x2_ic.t());

    // Project into camera images
    Vector2d nu1, nu2;
    cam.proj(p1_cl, nu1);
    cam.proj(p2_cl, nu2);

    // Project out of images
    Vector3d p1_cl_, p2_cl_;
    cam.invProj(nu1, p1_cl.norm(), p1_cl_);
    cam.invProj(nu2, p2_cl.norm(), p2_cl_);

    // Check the equation relating the pixels matched in each image
    Vector3d zeta1;
    Vector2d nu2_hat;
    double rho1 = 1.0 / p1_cl.norm();
    cam.invProj(nu1, 1.0, zeta1);
    Vector3d p2_cl_hat = x_bc.q_.rotp(x2_ib.q_.rotp(x1_ib.q_.rota(x_bc.q_.rota(1.0 / rho1 * zeta1) +
                         x_bc.t_) + x1_ib.t_ - x2_ib.t_) - x_bc.t_);
    cam.proj(p2_cl_hat, nu2_hat);

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
    xform::Xformd x_bc_hat = x_bc;
    x_bc_hat.t_ += noise;
    x_bc_hat.q_ += noise;

    // Initialize landmark inverse distances based on body positions
    vector<double> rho_hats;
    for (int i = 0; i < lm.cols(); ++i)
    {
      rho_hats.push_back(0.1);
    }

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
    {
      for (int j = 1; j < x_ibs.size(); ++j)
      {
        problem.AddResidualBlock(new FeatureFactor(
                                 new Feature(imgs[0][i], imgs[j][i], x_ibs[0], x_ibs[j], cam)),
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

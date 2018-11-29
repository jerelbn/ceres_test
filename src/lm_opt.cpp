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



void bodyPose(const double& t, xform::Xformd& x_ib)
{
  // Body pose as a function of time
  double x = 0;
  double y = sin(t);
  double z = cos(t);
  double phi = 0;
  double theta = 0;
  double psi = 0;

  // Replace elements
  x_ib.sett(Vector3d(x, y, z));
  x_ib.setq(quat::Quatd(phi, theta, psi));
}


int main()
{
  // Constants
//  srand((unsigned)time(NULL));

  // Build a camera
  Vector2d focal_len(483.4673, 481.3655);
  Vector2d cam_center(320.0, 240.0);
  Vector2d image_size(640.0, 480.0);
  Matrix<double, 5, 1> distortion;
  distortion << 0, 0, 0, 0, 0;
  double cam_skew(0);
  Camerad cam(focal_len, cam_center, distortion, cam_skew, image_size);

  // Landmark vectors in NED fixed frame
  Array<double, 3, 100> lm;
  lm.setRandom();
  lm.row(0) += 2;
  lm.row(1) *= 5;
  lm.row(2) *= 5;

  // Bady to camera translation and rotation
  Vector3d p_bc(0, 0, 0); // body to camera translation in body frame
  quat::Quatd q_bcb(0, 0, 0); // body to camera-body rotation
  quat::Quatd q_cbc(M_PI/2.0, 0.0, M_PI/2.0); // camera-body to camera rotation
  quat::Quatd q_bc = q_bcb * q_cbc; // body to camera rotation
  xform::Xformd x_bc(p_bc, q_bc);

  // Define position and attitude of two cameras in NED fixed frame
  xform::Xformd x1_ib, x2_ib;
  double t1 = 0.0;
  double t2 = 1.0;
  bodyPose(t1, x1_ib);
  bodyPose(t2, x2_ib);

  bool test_error = true;
  if (test_error)
  {
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
    Vector3d p2_cl_hat = x_bc.q().rotp(x2_ib.q().rotp(x1_ib.q().rota(x_bc.q().rota(1.0 / rho1 * zeta1) + x_bc.t()) + x1_ib.t() - x2_ib.t()) -x_bc.t());
    cam.proj(p2_cl_hat, nu2_hat);

    cout << "residual error = " << (nu2 - nu2_hat).transpose() << endl;
  }

  //

  return 0;
}

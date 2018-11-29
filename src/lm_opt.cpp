// Given body poses over time, compute camera extrinsic pose.
#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include "geometry/quat.h"
#include "geometry/support.h"
#include "geometry/cam.h"

using namespace std;
using namespace Eigen;


struct S3Plus
{
  template<typename T>
  bool operator()(const T* _q1, const T* _delta, T* _q2) const
  {
    quat::Quat<T> q1(_q1);
    Map<const Matrix<T,3,1>> delta(_delta);
    Map<Matrix<T,4,1>> q2(_q2);
    q2 = (q1 + delta).elements();
    return true;
  }
};


void bodyPose(const double& t, Vector3d& p_ib, quat::Quatd& q_ib)
{
  // Body pose as a function of time
  double x = 0;
  double y = sin(t);
  double z = cos(t);
  double phi = 0;
  double theta = 0;
  double psi = 0;

  // Replace elements
  p_ib = Vector3d(x, y, z);
  q_ib = quat::Quatd(phi, theta, psi);
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

  // Define position and attitude of two cameras in NED fixed frame
  Vector3d p1_ib, p2_ib;
  quat::Quatd q1_ib, q2_ib;
  double t1 = 0.0;
  double t2 = 1.0;
  bodyPose(t1, p1_ib, q1_ib);
  bodyPose(t2, p2_ib, q2_ib);

  Vector3d p1_ic = p1_ib + q1_ib.rota(p_bc);
  Vector3d p2_ic = p2_ib + q2_ib.rota(p_bc);
  quat::Quatd q1_ic = q1_ib * q_bc;
  quat::Quatd q2_ic = q2_ib * q_bc;

  // Get landmark vector in each camera frame
  Vector3d p1_cl = q1_ic.rotp(lm.col(0).matrix() - p1_ic);
  Vector3d p2_cl = q2_ic.rotp(lm.col(0).matrix() - p2_ic);

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
  Vector3d p2_cl_hat = q_bc.rotp(q2_ib.rotp(q1_ib.rota(q_bc.rota(1.0 / rho1 * zeta1) + p_bc) + p1_ib - p2_ib) - p_bc);
  cam.proj(p2_cl_hat, nu2_hat);

  cout << "residual error = " << (nu2 - nu2_hat).transpose() << endl;

//  // True rotation and translation direction from second to first camera
//  Vector3d t21 = (q2_i2c.rotp(Vector3d(p1_i2c - p2_i2c))).normalized();
//  quat::Quatd q21 = q2_i2c.inverse() * q1_i2c;
//  cout << "True rotation: " << q21.elements().transpose() << endl;
//  cout << "True translation direction: " << t21.transpose() << endl;

//  // Measurements in the first and second cameras
//  vector<Vector3d,aligned_allocator<Vector3d>> z1, z2;
//  for (int i = 0; i < lm.cols(); ++i)
//  {
//    z1.push_back((q1_i2c.rotp(Vector3d(lm.col(i).matrix() - p1_i2c))).normalized());
//    z2.push_back((q2_i2c.rotp(Vector3d(lm.col(i).matrix() - p2_i2c))).normalized());
//  }

//  // Initial guesses of R and t and initial errors
//  quat::Quatd q(0,0,0);
//  quat::Quatd qt(0,-1.5,0);
//  Vector3d q_err_init = quat::Quatd::log(q.inverse() * q21);
//  double t_err_init = vec_diff(t21, qt.uvec());

//  // Find R and t by nonlinear least squares
//  optimizePose(q, qt, z1, z2);

//  // Final errors
//  Vector3d q_err_final = quat::Quatd::log(q.inverse() * q21);
//  double t_err_final = vec_diff(t21, qt.uvec());

//  // Report data
//  cout << "Initial error (q,qt): " << q_err_init.norm() << ", " << t_err_init << endl;
//  cout << "Final error (q,qt):   " << q_err_final.norm() << ", " << t_err_final << endl;

  return 0;
}

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"

using namespace std;
using namespace Eigen;


// template<typename T>
// void qmul(const T* q1, const T* q2, T* q3)
// {
//   q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
//   q3[1] = q1[0]*q2[1] + q2[0]*q1[1] + q1[2]*q2[3] - q2[2]*q1[3];
//   q3[2] = q1[0]*q2[2] + q2[0]*q1[2] - q1[1]*q2[3] + q2[1]*q1[3];
//   q3[3] = q1[0]*q2[3] + q2[0]*q1[3] + q1[1]*q2[2] - q2[1]*q1[2];
// }


// template<typename T>
// void Rq(const T* q, Matrix<T,3,3>& R)
// {
//   // Pre-calculations
//   const T qw2 = q[0] * q[0];
//   const T qwqx = q[0] * q[1];
//   const T qwqy = q[0] * q[2];
//   const T qwqz = q[0] * q[3];
//   const T qxqy = q[1] * q[2];
//   const T qxqz = q[1] * q[3];
//   const T qyqz = q[2] * q[3];

//   // Output
//   R(0,0) = 2.0 * (qw2 + q[1] * q[1]) - 1.0;
//   R(0,1) = 2.0 * (qwqz + qxqy);
//   R(0,2) = 2.0 * (qxqz - qwqy);
//   R(1,0) = 2.0 * (qxqy - qwqz);
//   R(1,1) = 2.0 * (qw2 + q[2] * q[2]) - 1.0;
//   R(1,2) = 2.0 * (qwqx + qyqz);
//   R(2,0) = 2.0 * (qwqy + qxqz);
//   R(2,1) = 2.0 * (qyqz - qwqx);
//   R(2,2) = 2.0 * (qw2 + q[3] * q[3]) - 1.0;
// }


struct S3Plus
{
  template<typename T>
  bool operator()(const T* q1, const T* delta, T* q2) const
  {
    Map<const Matrix<T,3,1>> delta_(delta);
    common::Quaternion<T> q1_(q1[0], q1[1], q1[2], q1[3]);
    common::Quaternion<T> q2_ = q1_ + delta_;
    q2[0] = q2_.w();
    q2[1] = q2_.x();
    q2[2] = q2_.y();
    q2[3] = q2_.z();

//    const T delta2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
//    T dq[4];
//    if (delta2 > T(0.0))
//    {
//      T delta_norm = sqrt(delta2);
//      const T sn = sin(delta_norm) / delta_norm;
//      dq[0] = cos(delta_norm);
//      dq[1] = sn * delta[0];
//      dq[2] = sn * delta[1];
//      dq[3] = sn * delta[2];
//    }
//    else
//    {
//      dq[0] = T(1.0);
//      dq[1] = delta[0];
//      dq[2] = delta[1];
//      dq[3] = delta[2];
//    }
//    qmul(q1, dq, q2);
    return true;
  }
};


struct SampsonError
{
  SampsonError(const Vector3d& _t, const Vector3d& _e1, const Vector3d& _e2)
      : t(_t), e1(_e1), e2(_e2) {}

  template <typename T>
  bool operator()(const T* const q, T* residuals) const
  {
    static Matrix<T,3,3> R;
    common::Quaternion<T> q_(q[0], q[1], q[2], q[3]);
    R = q_.R();

    // Construct residual
    const Matrix<T,3,3> E = R * common::skew(Matrix<T,3,1>(t.cast<T>()));
    const Matrix<T,1,3> e1T_E = e1.cast<T>().transpose() * E;
    const Matrix<T,3,1> E_e2 = E * e2.cast<T>();
    const T e1T_E_e2 = e1.cast<T>().transpose() * E_e2;
    residuals[0] = e1T_E_e2 / sqrt(e1T_E(0) * e1T_E(0) + e1T_E(1) * e1T_E(1) + E_e2(0) * E_e2(0) + E_e2(1) * E_e2(1));
    return true;
  }

private:

  const Vector3d t, e1, e2;

};


// Relative pose optimizer using the Ceres Solver
void optimizePose(common::Quaterniond& q, const Vector3d& t,
                  const vector<Vector3d,aligned_allocator<Vector3d>>& e1,
                  const vector<Vector3d,aligned_allocator<Vector3d>>& e2,
                  const unsigned &iters)
{
  // Ensure number of directions vectors from each camera match
  if (e1.size() != e2.size())
  {
    std::cout << "\nError in optimizePose. Direction vector arrays must be the same size.\n\n";
    return;
  }

  // Build optimization problem with Ceres-Solver
  ceres::Problem problem;

  ceres::LocalParameterization *S3_local_parameterization =
    new ceres::AutoDiffLocalParameterization<S3Plus,4,3>;

  for (int i = 0; i < e1.size(); ++i)
  {
    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<SampsonError, 1, 4>(new SampsonError(t, e1[i], e2[i]));
    problem.AddResidualBlock(cost_function, NULL, q.data());
  }

  problem.SetParameterization(q.data(), S3_local_parameterization);

  // Solve for the optimal rotation and translation direciton
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = iters;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n\n";
}


int main()
{
  // Solver parameters
  const int num_iterations = 100;
   srand((unsigned)time(NULL));

  // Landmark parameters in NED fixed frame
  Array<double, 3, 100> lm;
  lm.setRandom();
  lm.row(0) += 10;
  lm.row(1) *= 20;
  lm.row(2) *= 20;

  // Bady to camera translation and rotation
  Vector3d pbc(0,0,0);
  common::Quaterniond qbc = common::Quaterniond::fromEuler(M_PI/2, 0.0, M_PI/2);

  // Define position and attitude of two cameras in NED fixed frame
  Vector3d p1_i2b(0,0,0);
  Vector3d p2_i2b(0,2,0);
  common::Quaterniond q1_i2b;
  common::Quaterniond q2_i2b = common::Quaterniond::fromEuler(0.0, 0.0, -0.1);

  Vector3d p1_i2c = p1_i2b + q1_i2b.rota(pbc);
  Vector3d p2_i2c = p2_i2b + q2_i2b.rota(pbc);
  common::Quaterniond q1_i2c = q1_i2b * qbc;
  common::Quaterniond q2_i2c = q2_i2b * qbc;

  // True rotation and translation direction from second to first camera
  Vector3d t21 = (q2_i2c.rotp(Vector3d(p1_i2c - p2_i2c))).normalized();
  common::Quaterniond q21 = q2_i2c.inverse() * q1_i2c;
//  cout << "True rotation: " << q21.toEigen().transpose() << endl;
//  cout << "True translation direction: " << t21.transpose() << endl;

  // Measurements in the first and second cameras
  vector<Vector3d,aligned_allocator<Vector3d>> z1, z2;
  for (int i = 0; i < lm.cols(); ++i)
  {
    z1.push_back((q1_i2c.rotp(Vector3d(lm.col(i).matrix() - p1_i2c))).normalized());
    z2.push_back((q2_i2c.rotp(Vector3d(lm.col(i).matrix() - p2_i2c))).normalized());
  }

  // Initial guesses of R and t and initial errors
  common::Quaterniond q = common::Quaterniond::fromEuler(0,0,0);
  Vector3d q_err_init = common::Quaterniond::log(q.inverse() * q21);

  // Find R and t by nonlinear least squares
  optimizePose(q, t21, z1, z2, num_iterations);

  // Final errors
  Vector3d q_err_final = common::Quaterniond::log(q.inverse() * q21);

  // Report data
  cout << "Initial rotation error: " << q_err_init.norm() << endl;
  cout << "Final rotation error:   " << q_err_final.norm() << endl;

//  // Rotation Testing
//  Matrix3d R1;
//  R1.setIdentity();
//  Vector3d delta(0.1,0.2,0.3);
//  Matrix3d R2;
//  R2.setZero();
//  cout << "R1 = \n" << R1 << endl;
//  ceres::AutoDiffLocalParameterization<SO3Plus,9,3> parameterization;
//  parameterization.Plus(R1.data(),delta.data(),R2.data());
//  cout << "R2 = \n" << R2 << endl;
//  Matrix<double, 9, 3> J;
//  parameterization.ComputeJacobian(R2.data(), J.data());
//  cout << "J = \n" << J << endl;

//  // Quaternion Testing
//  double q1[4] = {1,0,0,0};
//  Vector3d delta(0.1,0.2,0.3);
//  double q2[4] = {1,0,0,0};
//  cout << "q1 = [" << q1[0] << " " << q1[1] << " " << q1[2] << " " << q1[3] << "]\n";
//  ceres::AutoDiffLocalParameterization<S3Plus,4,3> parameterization;
//  parameterization.Plus(q1,delta.data(),q2);
//  cout << "q2 = [" << q2[0] << " " << q2[1] << " " << q2[2] << " " << q2[3] << "]\n";
//  double J[12];
//  parameterization.ComputeJacobian(q2,J);
//  Map<Matrix<double,4,3>> J_(J);
//  cout << J_ << endl;

  return 0;
}

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <chrono>
#include "common_cpp/quaternion.h"

using namespace std;
using namespace Eigen;


template<typename T>
T vec_diff(Matrix<T,3,1> v1, Matrix<T,3,1> v2)
{
  // make sure they are unit vectors
  v1.normalize();
  v2.normalize();

  // account for small numerical error just outside the range of cos^-1
  T v1T_v2 = v1.dot(v2);
  if (fabs(v1T_v2 - T(1.0)) < T(1e-12)) // same direction
    return 0;
  else if (fabs(v1T_v2 + T(1.0)) < T(1e-12)) // opposite direction
    return T(M_PI);
  else
    return acos(v1T_v2);
}


struct S3Plus
{
  template<typename T>
  bool operator()(const T* _q1, const T* _delta, T* _q2) const
  {
    common::Quaternion<T> q1(_q1);
    Map<const Matrix<T,3,1>> delta(_delta);
    Map<Matrix<T,4,1>> q2(_q2);
    q2 = (q1 + delta).toEigen();
    return true;
  }
};


struct S2Plus
{
  template<typename T>
  bool operator()(const T* _q1, const T* _delta, T* _q2) const
  {
    common::Quaternion<T> q1(_q1);
    Map<const Matrix<T,2,1>> delta(_delta);
    Map<Matrix<T,4,1>> q2(_q2);
    q2 = (common::Quaternion<T>::exp(q1.proj() * delta) * q1).toEigen();
    return true;
  }
};


struct SampsonError
{
  SampsonError(const Vector3d& _e1, const Vector3d& _e2)
      : e1(_e1), e2(_e2) {}

  template <typename T>
  bool operator()(const T* const _q, const T* _qt, T* residuals) const
  {
    // Map data
    common::Quaternion<T> q(_q);
    common::Quaternion<T> qt(_qt);

    // Construct residual
    Matrix<T,3,3> E = q.R() * common::skew(qt.uvec());
    Matrix<T,1,3> e1T_E = e1.cast<T>().transpose() * E;
    Matrix<T,3,1> E_e2 = E * e2.cast<T>();
    T e1T_E_e2 = e1.cast<T>().transpose() * E_e2;
    residuals[0] = e1T_E_e2 / sqrt(e1T_E(0) * e1T_E(0) + e1T_E(1) * e1T_E(1) + E_e2(0) * E_e2(0) + E_e2(1) * E_e2(1));
    return true;
  }

private:

  const Vector3d e1, e2;

};


// Relative pose optimizer using the Ceres Solver
void optimizePose(common::Quaterniond& q, common::Quaterniond& qt,
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

  // Does passing a dynamic rvalue result in a memory leak?
  problem.AddParameterBlock(q.data(), 4, new ceres::AutoDiffLocalParameterization<S3Plus,4,3>);
  problem.AddParameterBlock(qt.data(), 4, new ceres::AutoDiffLocalParameterization<S2Plus,4,2>);

  for (int i = 0; i < e1.size(); ++i)
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SampsonError, 1, 4, 4>
    (new SampsonError(e1[i], e2[i])), NULL, q.data(), qt.data());

  // Solve for the optimal rotation and translation direciton
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = iters;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n\n";
}


int main()
{
  // Solver parameters
  const int num_iterations = 20;
  srand((unsigned)time(NULL));

  // Landmark parameters in NED fixed frame
  Array<double, 3, 100> lm;
  lm.setRandom();
  lm.row(0) += 20;
  lm.row(1) *= 50;
  lm.row(2) *= 50;

  // Bady to camera translation and rotation
  Vector3d pbc(0,0,0);
  common::Quaterniond qbc(M_PI/2, 0.0, M_PI/2);

  // Define position and attitude of two cameras in NED fixed frame
  Vector3d p1_i2b(0,0,0);
  Vector3d p2_i2b(1.5,2,1);
  common::Quaterniond q1_i2b;
  common::Quaterniond q2_i2b(0.3, 0.2, -0.1);

  Vector3d p1_i2c = p1_i2b + q1_i2b.inv().rot(pbc);
  Vector3d p2_i2c = p2_i2b + q2_i2b.inv().rot(pbc);
  common::Quaterniond q1_i2c = q1_i2b * qbc;
  common::Quaterniond q2_i2c = q2_i2b * qbc;

  // True rotation and translation direction from second to first camera
  Vector3d t21 = (q2_i2c.rot(Vector3d(p1_i2c - p2_i2c))).normalized();
  common::Quaterniond q21 = q2_i2c.inv() * q1_i2c;
//  cout << "True rotation: " << q21.toEigen().transpose() << endl;
//  cout << "True translation direction: " << t21.transpose() << endl;

  // Measurements in the first and second cameras
  vector<Vector3d,aligned_allocator<Vector3d>> z1, z2;
  for (int i = 0; i < lm.cols(); ++i)
  {
    z1.push_back((q1_i2c.rot(Vector3d(lm.col(i).matrix() - p1_i2c))).normalized());
    z2.push_back((q2_i2c.rot(Vector3d(lm.col(i).matrix() - p2_i2c))).normalized());
  }

  // Initial guesses of R and t and initial errors
  common::Quaterniond q(0,0,0);
  common::Quaterniond qt(0,-1.5,0);
  Vector3d q_err_init = common::Quaterniond::log(q.inv() * q21);
  double t_err_init = vec_diff(t21, qt.uvec());

  // Find R and t by nonlinear least squares
  optimizePose(q, qt, z1, z2, num_iterations);

  // Final errors
  Vector3d q_err_final = common::Quaterniond::log(q.inv() * q21);
  double t_err_final = vec_diff(t21, qt.uvec());

  // Report data
  cout << "Initial error (q,qt): " << q_err_init.norm() << ", " << t_err_init << endl;
  cout << "Final error (q,qt):   " << q_err_final.norm() << ", " << t_err_final << endl;

//  Vector3d t = qt.uvec();
//  t(2) *= -1;
//  cout << "\n\n";
//  for (int i = 0; i < z1.size(); ++i)
//  {
//    Vector3d lm1 = z1[i];
//    Vector3d lm2 = z2[i];
//    Matrix3d E = q.R() * common::skew(t);
//    RowVector3d e1T_E = lm1.transpose() * E;
//    Vector3d E_e2 = E * lm2;
//    double e1T_E_e2 = lm1.transpose() * E_e2;
//    double sampson_error = e1T_E_e2 / sqrt(e1T_E(0) * e1T_E(0) + e1T_E(1) * e1T_E(1) + E_e2(0) * E_e2(0) + E_e2(1) * E_e2(1));
//    cout << "Sampson Error: " << sampson_error << endl;
//  }

  return 0;
}

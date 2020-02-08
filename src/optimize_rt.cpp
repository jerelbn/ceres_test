#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <chrono>
#include "common_cpp/quaternion.h"

using namespace std;
using namespace chrono;
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
    q2 = common::Quaternion<T>::boxPlusUnitVector(q1, delta).toEigen();
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
    T e1T_E_e2 = e1.cast<T>().dot(E_e2);
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


struct SampsonError2
{
  SampsonError2(const Vector3d& _e1, const Vector3d& _e2)
      : e1(_e1), e2(_e2) {}

  template <typename T>
  bool operator()(const T* const _q, const T* _qt, T* residuals) const
  {
    // Map data
    common::Quaternion<T> q(_q);
    common::Quaternion<T> qt(_qt);

    // Construct residual
    Matrix<T,3,3> E = common::skew(qt.uvec()) * q.R();
    Matrix<T,1,3> e2T_E = e2.cast<T>().transpose() * E;
    Matrix<T,3,1> E_e1 = E * e1.cast<T>();
    residuals[0] = e2.cast<T>().dot(E_e1) / sqrt(e2T_E(0) * e2T_E(0) + e2T_E(1) * e2T_E(1) + E_e1(0) * E_e1(0) + E_e1(1) * E_e1(1));
    return true;
  }

private:

  const Vector3d e1, e2;

};


// Relative pose optimizer using the Ceres Solver
void optimizePose2(common::Quaterniond& q, common::Quaterniond& qt,
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
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<SampsonError2, 1, 4, 4>
    (new SampsonError2(e1[i], e2[i])), NULL, q.data(), qt.data());

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


void sampson(double& S,
             const Vector3d& p1, const Vector3d& p2,
             const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  S = p2.dot(Ep1)/sqrt(p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1));
}


void sampsonDerivative(Matrix<double,1,5>& dS,
                       const Vector3d& p1, const Vector3d& p2,
                       const common::Quaterniond& q, const common::Quaterniond& qt)
{
  Matrix3d R = q.R();
  Vector3d et = qt.uvec();

  Matrix3d E = common::skew(et)*R;
  RowVector3d p2TE = p2.transpose()*E;
  Vector3d Ep1 = E*p1;

  double val = p2TE(0)*p2TE(0) + p2TE(1)*p2TE(1) + Ep1(0)*Ep1(0) + Ep1(1)*Ep1(1);
  double S = p2.dot(Ep1)/sqrt(val);
  
  Matrix3d dE_dq1 = -common::skew(et)*common::skew(common::e1)*R;
  Matrix3d dE_dq2 = -common::skew(et)*common::skew(common::e2)*R;
  Matrix3d dE_dq3 = -common::skew(et)*common::skew(common::e3)*R;
  Matrix3d dE_dqt1 = common::skew((qt.proj()*Vector2d(1,0)).cross(et))*R;
  Matrix3d dE_dqt2 = common::skew((qt.proj()*Vector2d(0,1)).cross(et))*R;

  dS(0) = (p2.dot(dE_dq1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq1.col(0)) + p2TE(1)*p2.dot(dE_dq1.col(1)) + Ep1(0)*(dE_dq1*p1)(0) + Ep1(1)*(dE_dq1*p1)(1)))/val;
  dS(1) = (p2.dot(dE_dq2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq2.col(0)) + p2TE(1)*p2.dot(dE_dq2.col(1)) + Ep1(0)*(dE_dq2*p1)(0) + Ep1(1)*(dE_dq2*p1)(1)))/val;
  dS(2) = (p2.dot(dE_dq3*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dq3.col(0)) + p2TE(1)*p2.dot(dE_dq3.col(1)) + Ep1(0)*(dE_dq3*p1)(0) + Ep1(1)*(dE_dq3*p1)(1)))/val;
  dS(3) = (p2.dot(dE_dqt1*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt1.col(0)) + p2TE(1)*p2.dot(dE_dqt1.col(1)) + Ep1(0)*(dE_dqt1*p1)(0) + Ep1(1)*(dE_dqt1*p1)(1)))/val;
  dS(4) = (p2.dot(dE_dqt2*p1)*sqrt(val) - S*(p2TE(0)*p2.dot(dE_dqt2.col(0)) + p2TE(1)*p2.dot(dE_dqt2.col(1)) + Ep1(0)*(dE_dqt2*p1)(0) + Ep1(1)*(dE_dqt2*p1)(1)))/val;
}




void sampsonDerivativeNumerical(Matrix<double,1,5>& dS,
                                const Vector3d& p1, const Vector3d& p2,
                                const common::Quaterniond& q, const common::Quaterniond& qt)
{
  static const double eps = 1e-6;
  double Sp, Sm;
  for (int i = 0; i < 3; ++i)
  {
    common::Quaterniond qp = q + eps * common::I_3x3.col(i);
    common::Quaterniond qm = q + -eps * common::I_3x3.col(i);
    sampson(Sp, p1, p2, qp, qt);
    sampson(Sm, p1, p2, qm, qt);
    dS(i) = (Sp - Sm)/(2.0*eps);
  }
  for (int i = 0; i < 2; ++i)
  {
    common::Quaterniond qtp = common::Quaterniond::boxPlusUnitVector(qt, eps*common::I_2x2.col(i));
    common::Quaterniond qtm = common::Quaterniond::boxPlusUnitVector(qt, -eps*common::I_2x2.col(i));
    sampson(Sp, p1, p2, q, qtp);
    sampson(Sm, p1, p2, q, qtm);
    dS(i+3) = (Sp - Sm)/(2.0*eps);
  }
}


void sampsonLM(common::Quaterniond& q, common::Quaterniond& qt,
               const vector<Vector3d,aligned_allocator<Vector3d>>& e1,
               const vector<Vector3d,aligned_allocator<Vector3d>>& e2,
               const int& max_iters=50, const double& exit_tol=1e-6,
               const double& lambda0=1, const double& lambda_adjust=10)
{
  double lambda = lambda0;

  common::Quaterniond q_new, qt_new;
  Matrix<double,1,5> dS;
  Matrix<double,5,1> b, delta;
  Matrix<double,5,5> H, H_diag, A;
  unsigned N = e1.size();
  VectorXd cost(N), cost_new(N);
  MatrixXd J(N,5);
  double cost_squared;
  bool prev_fail = false;
  for (int i = 0; i < max_iters; ++i)
  {
    if (!prev_fail)
    {
      // Build cost function and Jacobian
      for (int j = 0; j < N; ++j)
      {
        sampson(cost(j), e1[j], e2[j], q, qt);
        sampsonDerivative(dS, e1[j], e2[j], q, qt);
        // sampsonDerivativeNumerical(dS, e1[j], e2[j], q, qt);
        J.row(j) = dS;
      }
      H = J.transpose()*J;
      b = -J.transpose()*cost;
      cost_squared = cost.dot(cost);
    }
    H_diag = H.diagonal().asDiagonal();
    A = H + lambda*H_diag;
    delta = A.householderQr().solve(b);

    // Compute cost with new parameters
    q_new = q + delta.head<3>();
    qt_new = common::Quaterniond::boxPlusUnitVector(qt, delta.tail<2>());
    for (int j = 0; j < N; ++j)
      sampson(cost_new(j), e1[j], e2[j], q_new, qt_new);
    if (cost_new.dot(cost_new) < cost_squared)
    {
      q = q_new;
      qt = qt_new;
      lambda /= lambda_adjust;
      prev_fail = false;
    }
    else
    {
      lambda *= lambda_adjust;
      prev_fail = true;
    }

    if (delta.norm() < exit_tol) break;
  }
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
  common::Quaterniond qbc = common::Quaterniond::fromEuler(M_PI/2, 0.0, M_PI/2);

  // Define position and attitude of two cameras in NED fixed frame
  Vector3d p1_i2b(0,0,0);
  Vector3d p2_i2b(1.5,2,1);
  common::Quaterniond q1_i2b;
  common::Quaterniond q2_i2b = common::Quaterniond::fromEuler(0.3, 0.2, -0.1);

  Vector3d p1_i2c = p1_i2b + q1_i2b.inverse().rotp(pbc);
  Vector3d p2_i2c = p2_i2b + q2_i2b.inverse().rotp(pbc);
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
  common::Quaterniond qt = common::Quaterniond::fromUnitVector(Vector3d(-1,0,0));
  Vector3d q_err_init = common::Quaterniond::log(q.inverse() * q21);
  double t_err_init = vec_diff(t21, qt.uvec());

  common::Quaterniond q0 = q;
  common::Quaterniond qt0 = qt;

  auto t_calc_0 = high_resolution_clock::now();

  // Find R and t by nonlinear least squares
  // optimizePose(q, qt, z1, z2, num_iterations);

  // q = q.inverse();
  // optimizePose2(q, qt, z1, z2, num_iterations);
  // q = q.inverse();

  q = q.inverse();
  sampsonLM(q, qt, z1, z2, num_iterations, 1e-6, 1e-6, 10);
  q = q.inverse();

  double dt_calc = duration_cast<microseconds>(high_resolution_clock::now() - t_calc_0).count()*1e-6;

  // Final errors
  Vector3d q_err_final = common::Quaterniond::log(q.inverse() * q21);
  double t_err_final = vec_diff(t21, qt.uvec());

  // Report data
  cout << "Calc time taken: " << dt_calc << " seconds\n";
  cout << "Initial error (q,qt): (" << q_err_init.norm() << ", " << t_err_init << ")" << endl;
  cout << "Final error (q,qt):   (" << q_err_final.norm() << ", " << t_err_final << ")" << endl;
  cout << "q0 = " << q0.toEigen().transpose() << endl;
  cout << "qf = " << q.toEigen().transpose() << endl;
  cout << "q_true = " << q21.toEigen().transpose() << endl;
  cout << "t0 = " << qt0.uvec().transpose() << endl;
  cout << "tf = " << qt.uvec().transpose() << endl;
  cout << "t_true = " << common::Quaterniond::fromUnitVector(t21).uvec().transpose() << endl;

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

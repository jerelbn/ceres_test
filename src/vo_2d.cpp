// A simple 1D example of odometry where we measure velocity at each time
// step and also the intial position (to constrain it to the truth)

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;


struct PositionFactor
{
  PositionFactor(const Vector2d& p_meas)
      : p_meas_(p_meas) {}

  template <typename T>
  bool operator()(const T* const x, T* residuals) const
  {
    Map<const Matrix<T,2,1>> p(x);
    Map<Matrix<T,2,1>> p_err(residuals);
    p_err = p - p_meas_.cast<T>();
    return true;
  }

private:

  const Vector2d p_meas_;

};


struct OdomFactor
{
  OdomFactor(const Vector2d& v_meas, const double& dt)
      : v_meas_(v_meas), dt_(dt) {}

  template <typename T>
  bool operator()(const T* const x1, const T* const x2, T* residuals) const
  {
    Map<const Matrix<T,2,1>> p1(x1);
    Map<const Matrix<T,2,1>> p2(x2);
    Map<Matrix<T,2,1>> p_err(residuals);
    p_err = (p1 + v_meas_.cast<T>() * T(dt_)) - p2;
    return true;
  }

private:

  const Vector2d v_meas_;
  const double dt_;

};


int main()
{
  // Create truth
  vector<Vector2d, aligned_allocator<Vector2d>> truth;
  double tf = 1.0;
  double dt = 0.1;
  Vector2d v_true(0.1, 0.2);
  auto true_pos = [](const double& t, const Vector2d& v){ return v * t; };
  double t = 0;
  while (t <= tf)
  {
    truth.push_back(true_pos(t, v_true));
    t += dt;
  }

  // Initialize estimate
  vector<Vector2d, aligned_allocator<Vector2d>> est(truth.size());
  Vector2d est0(0.5, 0.6);
  for (int i = 0; i < est.size(); ++i)
    est[i] = est0 * (i+1);

  // Compare initial estimate with truth
  cout << "\ni\test\t\ttruth" << endl;
  for (int i = 0; i < est.size(); ++i)
    cout << i << "\t" << est[i].transpose() << "\t\t" << truth[i].transpose() << endl;
  cout << endl;

  // Build optimization problem with Ceres-Solver
  ceres::Problem problem;

  // Add parameter blocks
  for (int i = 0; i < est.size(); ++i)
    problem.AddParameterBlock(est[i].data(), 2);

  // Add position factor
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<PositionFactor, 2, 2>(new PositionFactor(truth[0]));
  problem.AddResidualBlock(cost_function, NULL, est[0].data());

  // Add odometry factor
  for (int i = 0; i < est.size()-1; ++i)
  {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<OdomFactor, 2, 2, 2>(new OdomFactor(v_true, dt));
    problem.AddResidualBlock(cost_function, NULL, est[i].data(), est[i+1].data());
  }

  // Solve for the optimal state
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 50;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n\n";

  // Compare final estimate with truth
  cout << "\ni\test\t\ttruth" << endl;
  for (int i = 0; i < est.size(); ++i)
    cout << i << "\t" << est[i].transpose() << "\t\t" << truth[i].transpose() << endl;
  cout << endl;
}

// A simple 1D example of odometry where we measure velocity at each time
// step and also the intial position (to constrain it to the truth)

#include <ceres/ceres.h>
#include <iostream>
#include <random>

using namespace std;


struct PositionFactor
{
  PositionFactor(const double& p_meas)
      : p_meas_(p_meas) {}

  template <typename T>
  bool operator()(const T* const x, T* residuals) const
  {
    residuals[0] = x[0] - p_meas_;
    return true;
  }

private:

  const double p_meas_;

};


struct OdomFactor
{
  OdomFactor(const double& v_meas, const double& dt, const int& idx)
      : v_meas_(v_meas), idx_(idx), dt_(dt) {}

  template <typename T>
  bool operator()(const T* const x, T* residuals) const
  {
    residuals[0] = (x[idx_] + v_meas_ * dt_) - x[idx_+1];
    return true;
  }

private:

  const double v_meas_, dt_;
  const int idx_;

};


int main()
{
  // Create truth
  vector<double> truth;
  double tf = 1.0;
  double dt = 0.1;
  double v_true = 0.1;
  auto true_pos = [](const double& t){ return 0.1 * t; };
  double t = 0;
  while (t <= tf)
  {
    truth.push_back(true_pos(t));
    t += dt;
  }

  // Initialize estimate
  vector<double> est(truth.size());
  double est0 = 0.5;
  for (int i = 0; i < est.size(); ++i)
    est[i] = est0 * (i+1);

  // Compare initial estimate with truth
  cout << "\ni\test\ttruth" << endl;
  for (int i = 0; i < est.size(); ++i)
    cout << i << "\t" << est[i] << "\t" << truth[i] << endl;
  cout << endl;

  // Random noise for position and velocity measurements
  default_random_engine rng((unsigned)time(NULL));
  normal_distribution<double> dist(0, 0.01);

  // Build optimization problem with Ceres-Solver
  ceres::Problem problem;

  // Add parameter blocks
  problem.AddParameterBlock(est.data(), 11);

  // Add position factor
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<PositionFactor, 1, 11>(new PositionFactor(truth[0]));
  problem.AddResidualBlock(cost_function, NULL, est.data());

  // Add odometry factor
  for (int i = 0; i < est.size()-1; ++i)
  {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<OdomFactor, 1, 11>(new OdomFactor(v_true + dist(rng), dt, i));
    problem.AddResidualBlock(cost_function, NULL, est.data());
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
  cout << "i\test\ttruth" << endl;
  for (int i = 0; i < est.size(); ++i)
    cout << i << "\t" << est[i] << "\t" << truth[i] << endl;
  cout << endl;
}

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

struct CostFunctor
{
  template<typename T>
  bool operator()(const T* const x, T* residual) const
  {
    // Create templated unit vectors
    static Eigen::Matrix<T,2,1> e1 = Eigen::Vector2d(1,0).cast<T>();
    static Eigen::Matrix<T,2,1> e2 = Eigen::Vector2d(0,1).cast<T>();

    // Map input pointer to Eigen
    Eigen::Map<const Eigen::Matrix<T,2,1>> x_(x);

    // Construct the residual
    T f1 = 10.0 * (e2.dot(x_) - e1.dot(x_) * e1.dot(x_));
    T f2 = 1.0 - e1.dot(x_);
    residual[0] = 0.5 * (f1 * f1 + f2 * f2);
    return true;
  }
};


int main()
{
  // The variable to solve for with its initial value.
  double initial_x1 = 2;
  double initial_x2 = 2;
  Eigen::Vector2d x(initial_x1, initial_x2);

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 2>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, x.data());

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.function_tolerance = 1e-5;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x1 << ", " << initial_x2
            << " -> " << x[0] << ", " << x[1] << "\n";
  return 0;
}

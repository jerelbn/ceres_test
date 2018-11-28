#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

bool use_analytic_cost = true;
double a = 1.0;
double b = 100.0;

struct CostFunctor
{
  template<typename T>
  bool operator()(const T* const x, T* residual) const
  {
    T f1 = T(a) - x[0];
    T f2 = sqrt(T(b)) * (x[1] - x[0] * x[0]);
    residual[0] = f1 * f1 + f2 * f2;
    return true;
  }
};


struct RosenbrockAnalytic : public ceres::SizedCostFunction<1,2>
{
  RosenbrockAnalytic() {}
  virtual ~RosenbrockAnalytic() {}
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
  {
    const double x0 = parameters[0][0];
    const double x1 = parameters[0][1];

    double f1 = a - x0;
    double f2 = sqrt(b) * (x1 - x0 * x0);
    residuals[0] = f1 * f1 + f2 * f2;

    if (!jacobians) return true;
    double* jacobian = jacobians[0];
    if (!jacobian) return true;

    jacobian[0] = -2.0 * (a - x0) - 4.0 * x0 * b * (x1 - x0 * x0);
    jacobian[1] = 2.0 * b * (x1 - x0 * x0);
    return true;
  }

};


int main()
{
  // The variable to solve for with its initial value.
  double initial_x1 = 2.0;
  double initial_x2 = 2.0;
  double x[2] = {initial_x1, initial_x2};

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  ceres::CostFunction* cost_function;
  if (use_analytic_cost)
    cost_function = new RosenbrockAnalytic();
  else
    cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 2>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, x);

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10000;
  options.gradient_tolerance = 1e-9;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x0 : " << initial_x1 << ", " << initial_x2 << "\n";
  std::cout << "x  : " << x[0] << ", " << x[1] << "\n";

  return 0;
}

// A simple 1D example of odometry where we measure velocity at each time
// step and also the intial position (to constrain it to the truth)

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;


Matrix3d analytical_jacobian(const Vector3d& x)
{
  Matrix3d J;
  J <<    -1,  1,    -1,
       -x(2), -1, -x(0),
          -1,  0,     0;
  return J;
}

void analytical_jacobian(const double x[3], double J[3][3])
{
  J[0][0] = -1;
  J[0][1] = 1;
  J[0][2] = -1;
  J[1][0] = -x[2];
  J[1][1] = -1;
  J[1][2] = -x[0];
  J[2][0] = -1;
  J[2][1] = 0;
  J[2][2] = 0;
}


struct StateDerivative {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = -x[0] + x[1] - x[2];
    residual[1] = -x[0]*x[2] - x[1];
    residual[2] = -x[0];
    return true;
  }
};


int main()
{
  default_random_engine rng(time(NULL));
  uniform_real_distribution<double> dist(-10,10);
  double x1 = dist(rng);
  double x2 = dist(rng);
  double x3 = dist(rng);
  printf("x = \n%10.6f\n%10.6f\n%10.6f\n\n", x1, x2, x3);

  /*============= With Eigen ===============*/
  {
    Vector3d x(x1, x2, x3); // Create state
    Matrix3d J_analytical = analytical_jacobian(x);

    // Autodiff Jacobian using Eigen
    double const* x1_ptr_ptr[1]{x.data()};
    Vector3d r;
    Matrix<double,3,3,RowMajor> J_autodiff;
    double* J1_ptr_ptr[1]{J_autodiff.data()};
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<StateDerivative, 3, 3>(new StateDerivative());
    cost_function->Evaluate(x1_ptr_ptr, r.data(), J1_ptr_ptr);
    // NOTE: Evaluate gives the Jacobian of the i^th parameter block in row-major form
    // and since x is a single parameter block, we only have one row of in the Jacobian

    // Prints
    cout << "J_analytical = \n" << J_analytical << endl;
    cout << "\nJ_autodiff = \n" << J_autodiff << endl;
    cout << "\nr = \n" << r << endl;
  }


  /*============= Without Eigen ===============*/
  {
    double x[3]{x1, x2, x3}; // Create state
    double J_analytical[3][3];
    analytical_jacobian(x, J_analytical);

    // Autodiff Jacobian
    double const* x1_ptr_ptr[1]{x};
    double r[3];
    double* J_autodiff[1]{new double[9]};
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<StateDerivative, 3, 3>(new StateDerivative());
    cost_function->Evaluate(x1_ptr_ptr, r, J_autodiff);

    // Prints
    printf("\nJ_analytical = \n%10.6f\t%10.6f\t%10.6f\n%10.6f\t%10.6f\t%10.6f\n%10.6f\t%10.6f\t%10.6f\n\n",
           J_analytical[0][0], J_analytical[0][1], J_analytical[0][2],
           J_analytical[1][0], J_analytical[1][1], J_analytical[1][2],
           J_analytical[2][0], J_analytical[2][1], J_analytical[2][2]);
    printf("J_autodiff = \n%10.6f\t%10.6f\t%10.6f\n%10.6f\t%10.6f\t%10.6f\n%10.6f\t%10.6f\t%10.6f\n\n",
            J_autodiff[0][0], J_autodiff[0][1], J_autodiff[0][2],
            J_autodiff[0][3], J_autodiff[0][4], J_autodiff[0][5],
            J_autodiff[0][6], J_autodiff[0][7], J_autodiff[0][8]);
    printf("r = \n%10.6f\n%10.6f\n%10.6f\n\n", r[0], r[1], r[2]);
  }
}

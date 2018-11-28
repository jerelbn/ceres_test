// Jacobian of a typical state involving a rigid body and IMU biases

#include "common_cpp/common.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;

#define THETA1 2.0


enum {PX, PY, PZ, VX, VY, VZ, QW, QX, QY, QZ, AX, AY, AZ, GX, GY, GZ, CX, CY, STATE_SIZE};
enum {DPX, DPY, DPZ, DVX, DVY, DVZ, DQX, DQY, DQZ, DAX, DAY, DAZ, DGX, DGY, DGZ, DCX, DCY, DELTA_STATE_SIZE};
typedef Matrix<double, STATE_SIZE, 1> State;
typedef Matrix<double, DELTA_STATE_SIZE, 1> DeltaState;


// Local parameterization for Quaternions
struct StatePlus
{
  template<typename T>
  bool operator()(const T* x1, const T* delta, T* x2) const
  {
    Map<const Matrix<T,3,1>> p(x1+PX);
    Map<const Matrix<T,3,1>> v(x1+VX);
    const common::Quaternion<T> q(x1+QW);
    Map<const Matrix<T,3,1>> ba(x1+AX);
    Map<const Matrix<T,3,1>> bg(x1+GX);
    Map<const Matrix<T,2,1>> mu(x1+CX);
    Map<const Matrix<T,DELTA_STATE_SIZE,1>> delta_(delta);
    Map<Matrix<T,STATE_SIZE,1>> x2_(x2);
    x2_.template segment<3>(PX) = p + delta_.template segment<3>(DPX);
    x2_.template segment<3>(VX) = v + delta_.template segment<3>(DVX);
    x2_.template segment<4>(QW) = (q + Matrix<T,3,1>(delta_.template segment<3>(DQX))).toEigen();
    x2_.template segment<3>(AX) = ba + delta_.template segment<3>(DAX);
    x2_.template segment<3>(GX) = bg + delta_.template segment<3>(DGX);
    x2_.template segment<2>(CX) = mu + delta_.template segment<2>(DCX);
    return true;
  }
};


// Make the cost function the state dynamics and use Ceres to compute automatic derivative
struct StateDerivative
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  StateDerivative(const Vector3d& _acc, const Vector3d& _omega)
    : acc(_acc), omega(_omega) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const
  {
    // Constants
    static Matrix<T,3,3> E3(common::e3.cast<T>() * common::e3.transpose().cast<T>());
    static Matrix<T,3,3> IE3(common::I_3x3.cast<T>() - common::e3.cast<T>() * common::e3.transpose().cast<T>());
    static Matrix<T,3,2> I_3x2(common::I_2x3.transpose().cast<T>());
    static Matrix<T,3,1> e3(common::e3.cast<T>());
    static T g = T(common::gravity);

    // Map states
    Map<const Matrix<T,3,1>> v(x+VX);
    const common::Quaternion<T> q(x+QW);
    Map<const Matrix<T,3,1>> ba(x+AX);
    Map<const Matrix<T,3,1>> bg(x+GX);
    Map<const Matrix<T,2,1>> mu(x+CX);

    // Compute output
    Map<Matrix<T,DELTA_STATE_SIZE,1>> dx(residual);
    dx.setZero();
    dx.template segment<3>(DPX) = q.inv().rot(v);
    dx.template segment<3>(DVX) = E3 * (acc.cast<T>() - ba) + g * q.rot(e3) - (omega.cast<T>() - bg).cross(v) -
                                  Matrix<T,3,3>(v.asDiagonal()) * Matrix<T,3,3>(v.asDiagonal()) * I_3x2 * mu;
    dx.template segment<3>(DQX) = omega.cast<T>() - bg;
    return true;
  }

private:

  const Vector3d acc, omega;

};


Matrix<double,DELTA_STATE_SIZE,DELTA_STATE_SIZE> analytical_jacobian(const State& x, const Vector3d& acc, const Vector3d& omega)
{
  // Constants
  static Matrix<double,3,2> I_3x2(common::I_2x3.transpose());

  // Unpack state
  Vector3d v(x.segment<3>(VX));
  common::Quaterniond q(x.segment<4>(QW));
  Vector3d ba(x.segment<3>(AX));
  Vector3d bg(x.segment<3>(GX));
  Vector2d mu(x.segment<2>(CX));

  // Pack output
  Matrix<double,DELTA_STATE_SIZE,DELTA_STATE_SIZE> J;
  J.setZero();
  J.block<3,3>(DPX,DVX) = q.inv().R();
  J.block<3,3>(DPX,DQX) = -q.inv().R() * common::skew(v);
  J.block<3,3>(DVX,DVX) = -common::skew(Vector3d(omega - bg)) - 2.0 * Matrix3d((I_3x2 * mu).asDiagonal()) * Matrix3d(v.asDiagonal());
  J.block<3,3>(DVX,DQX) = common::gravity * common::skew(Vector3d(q.rot(common::e3)));
  J.block<3,3>(DVX,DAX) = -common::e3 * common::e3.transpose();
  J.block<3,3>(DVX,DGX) = -common::skew(v);
  J.block<3,2>(DVX,DCX) = -Matrix3d(v.asDiagonal()) * Matrix3d(v.asDiagonal()) * I_3x2;
  J.block<3,3>(DQX,DGX) = -common::I_3x3;
  return J;
}


int main()
{
  srand((unsigned)time(NULL));
  State x;
  x.setRandom();
  x.segment<4>(QW).normalize();
  Vector3d acc, omega;
  acc.setRandom();
  omega.setRandom();

  // Analytical Jacobian
  Matrix<double,DELTA_STATE_SIZE,DELTA_STATE_SIZE> J_analytical = analytical_jacobian(x, acc, omega);

  // Autodiff Jacobian using Eigen
  Matrix<double,DELTA_STATE_SIZE,STATE_SIZE,RowMajor> J_autodiff;
  DeltaState r;
  double const* x1_ptr_ptr[1]{x.data()};
  double* J1_ptr_ptr[1]{J_autodiff.data()};

//  ceres::Problem problem; // Need to create a problem to set the local parameterization
//  ceres::LocalParameterization *state_local_parameterization =
//      new ceres::AutoDiffLocalParameterization<StatePlus,STATE_SIZE,DELTA_STATE_SIZE>;
//  problem.AddParameterBlock(x.data(), STATE_SIZE, state_local_parameterization);
  ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<StateDerivative, DELTA_STATE_SIZE, STATE_SIZE>(new StateDerivative(acc, omega));
  cost_function->Evaluate(x1_ptr_ptr, r.data(), J1_ptr_ptr);
  // NOTE: Evaluate gives the Jacobian of the i^th parameter block in row-major form
  // and since x is a single parameter block, we only have one row of in the Jacobian

  // Compute derivative of q plus delta w.r.t. delta
  Matrix<double,4,3> dqplusdelta_ddelta;
  dqplusdelta_ddelta << -x(QX), -x(QY), -x(QZ),
                         x(QW), -x(QZ),  x(QY),
                         x(QZ),  x(QW), -x(QX),
                        -x(QY),  x(QX),  x(QW);
  dqplusdelta_ddelta *= 0.5;

  // Compute derivative of q plus delta w.r.t. delta
  Matrix<double,STATE_SIZE,DELTA_STATE_SIZE> J_delta;
  J_delta.setZero();
  J_delta.block<3,3>(PX,DPX).setIdentity();
  J_delta.block<3,3>(VX,DVX).setIdentity();
  J_delta.block<4,3>(QW,DQX) = dqplusdelta_ddelta;
  J_delta.block<3,3>(AX,DAX).setIdentity();
  J_delta.block<3,3>(GX,DGX).setIdentity();
  J_delta.block<2,2>(CX,DCX).setIdentity();

  // Compute minimal Jacobian of the state dynamics
  Matrix<double,DELTA_STATE_SIZE,DELTA_STATE_SIZE> J = J_autodiff * J_delta;

  // Prints
  cout << "x = \n" << x << "\n\n";
  cout << "r = \n" << r << "\n\n";
  cout << "J_autodiff = \n" << J_autodiff << "\n\n";
  cout << "J_delta = \n" << J_delta << "\n\n";
  cout << "J_analytical = \n" << J_analytical << "\n\n";
  cout << "J = \n" << J << "\n\n";
  cout << "J_error_norm = " << (J_analytical - J).norm() << endl;
}

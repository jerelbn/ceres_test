 // Jacobian of a typical state involving a rigid body and IMU biases

#include "common_cpp/common.h"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;


// Local parameterization for Quaternions representing rotations
struct S3Plus
{
  template<typename T>
  bool operator()(const T* _q, const T* _delta, T* _q_plus_delta) const
  {
    const common::Quaternion<T> q(_q);
    Map<const Matrix<T,3,1>> delta(_delta);
    Map<Matrix<T,4,1>> q_plus_delta(_q_plus_delta);
    q_plus_delta = (q + delta).toEigen();
    return true;
  }
};


// Local parameterization for Quaternions representing unit vectors
struct S2Plus
{
  template<typename T>
  bool operator()(const T* _q1, const T* _delta, T* _q2) const
  {
    const common::Quaternion<T> q1(_q1);
    Map<const Matrix<T,2,1>> delta(_delta);
    Map<Matrix<T,4,1>> q2(_q2);
    q2 = (common::Quaternion<T>::boxplus_uvec(q1, delta)).toEigen();
    return true;
  }
};


// Derivative of q + delta w.r.t. delta
template<typename T>
Matrix<T,4,3> dqpd_dd(const Matrix<T,4,1>& q)
{
  Matrix<T,4,3> m;
  m << -q(1), -q(2), -q(3),
        q(0), -q(3),  q(2),
        q(3),  q(0), -q(1),
       -q(2),  q(1),  q(0);
  m *= 0.5;
  return m;
}


// Derivative of dq w.r.t. q
template<typename T>
Matrix<T,3,4> ddq_dq(const Matrix<T,4,1>& q)
{
  Matrix<T,3,4> m;
  m << -q(1),  q(0),  q(3), -q(2),
       -q(2), -q(3),  q(0),  q(1),
       -q(3),  q(2), -q(1),  q(0);
  m *= 2.0;
  return m;
}


// Derivative of qt + delta w.r.t. delta
template<typename T>
Matrix<T,4,2> dqtpd_dd(const Matrix<T,4,1>& q)
{
  Matrix<T,4,2> m;
  double aa = 0.5 - q(0) * q(0) - q(1) * q(1) - q(2) * q(2) - q(3) * q(3);
  m <<   q(1) * aa,  q(2) * aa,
        -q(0) * aa,  -0.5 * q(3),
        0.5 * q(3), -q(0) * aa,
       -0.5 * q(2),   0.5 * q(1);
  return m;
}


// Derivative of dqt w.r.t. q
template<typename T>
Matrix<T,2,4> ddqt_dq(const Matrix<T,4,1>& q)
{
  return T(4.0) * dqtpd_dd(q).transpose();
}


// Make the cost function the state dynamics and use Ceres to compute automatic derivative
struct H_VO
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  H_VO(const common::Quaterniond& _qb)
    : qb(_qb) {}

  template <typename T>
  bool operator()(const T* const _qa, T* residual) const
  {
    const common::Quaternion<T> qa(_qa);
    Map<Matrix<T,4,1>> r(residual);
    r = (qa * qb.cast<T>()).toEigen();
    return true;
  }

private:

  const common::Quaterniond qb;

};


// Make the cost function the quaternion and use Ceres to compute automatic derivative
struct H2
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  H2(const common::Quaterniond& _q_bc, const Vector3d& _p_nb, const Vector3d& _p_bk)
    : q_bc(_q_bc), p_bk(_p_bk), p_nb(_p_nb) {}

  template <typename T>
  bool operator()(const T* const _q_nb, T* residual) const
  {
    const common::Quaternion<T> q_nb(_q_nb);
    Map<Matrix<T,4,1>> r(residual);
    Matrix<T,3,1> pt = q_bc.cast<T>().rot(q_nb.rot(p_bk.cast<T>() - p_nb.cast<T>()));
    r = (common::Quaternion<T>(pt)).toEigen();
    return true;
  }

private:

  const common::Quaterniond q_bc;
  const Vector3d p_bk, p_nb;

};


int main()
{
  srand(unsigned(time(nullptr)));
  bool rotations = false;
  double eps = 1e-5;

  if (rotations)
  {
    // Model
    Vector4d a, b;
    common::Quaterniond qa(a.setRandom().normalized());
    common::Quaterniond qb(b.setRandom().normalized());
    common::Quaterniond h = qa * qb;

    // Analytical Jacobian
    Matrix3d J_analytical = qb.R();

    // Numerical Jacobian
    Matrix3d J_numerical;
    J_numerical.setZero();
    for (int i = 0; i < 3; ++i)
    {
      Vector3d delta = Vector3d::Zero();
      delta(i) = eps;
      common::Quaterniond qap = qa + delta;
      common::Quaterniond qam = qa + -delta;
      common::Quaterniond hp = qap * qb;
      common::Quaterniond hm = qam * qb;
      J_numerical.col(i) = (hp - hm) / (2.0 * eps);
    }

    // Autodiff Jacobian using Eigen
    Matrix<double,4,4,RowMajor> J_autodiff;
    Vector4d r;
    double const* qa_ptr_ptr[1]{qa.data()};
    double* J_ptr_ptr[1]{J_autodiff.data()};

    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<H_VO, 4, 4>(new H_VO(qb));
    cost_function->Evaluate(qa_ptr_ptr, r.data(), J_ptr_ptr);
    // NOTE: Evaluate gives the Jacobian of the i^th parameter block in row-major form
    // and since x is a single parameter block, we only have one row of in the Jacobian

    // Compute derivatives
    Matrix<double,4,3> J_delta = dqpd_dd(qa.toEigen());
    Matrix<double,3,4> J_q = ddq_dq(h.toEigen());

    // Compute minimal Jacobian of the state dynamics
    Matrix<double,4,3> J_autodiff_delta = J_autodiff * J_delta;
    Matrix3d J = J_q * J_autodiff_delta;

    // Prints
    cout << "r = \n" << r << "\n\n";
    cout << "J_autodiff = \n" << J_autodiff << "\n\n";
    cout << "J_delta = \n" << J_delta << "\n\n";
    cout << "J_autodiff_delta = \n" << J_autodiff_delta << "\n\n";
    cout << "J = \n" << J << "\n\n";
    cout << "J_analytical = \n" << J_analytical << "\n\n";
    cout << "J_numerical = \n" << J_numerical << "\n\n";
  }
  else
  {
    // Maybe do the derivative of pt w.r.t. position or whatever
    Vector4d a;
    Vector3d b;
    common::Quaterniond q_bc(a.setRandom().normalized());
    common::Quaterniond q_bk(a.setRandom().normalized());
    common::Quaterniond q_nb(a.setRandom().normalized());
    Vector3d p_bk = Vector3d::Random();
    Vector3d p_nb = Vector3d::Random();
    Vector3d pt = q_bc.rot(q_nb.rot(p_bk - p_nb));
    common::Quaterniond h(pt);

    // Numerical Jacobian
    Matrix<double,2,3> J_numerical;
    J_numerical.setZero();
    for (int i = 0; i < 3; ++i)
    {
      Vector3d delta = Vector3d::Zero();
      delta(i) = eps;
      common::Quaterniond qp = q_nb + delta;
      common::Quaterniond qm = q_nb + -delta;
      Vector3d ptp = q_bc.rot(qp.rot(p_bk - p_nb));
      Vector3d ptm = q_bc.rot(qm.rot(p_bk - p_nb));
      common::Quaterniond hp(ptp);
      common::Quaterniond hm(ptm);
      J_numerical.col(i) = common::Quaterniond::log_uvec(hp, hm) / (2.0 * eps);
    }

    // Autodiff Jacobian using Eigen
    Matrix<double,4,4,RowMajor> J_autodiff;
    Vector4d r;
    double const* q_nb_ptr_ptr[1]{q_nb.data()};
    double* J_ptr_ptr[1]{J_autodiff.data()};

    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<H2, 4, 4>(new H2(q_bc, p_nb, p_bk));
    cost_function->Evaluate(q_nb_ptr_ptr, r.data(), J_ptr_ptr);
    // NOTE: Evaluate gives the Jacobian of the i^th parameter block in row-major form
    // and since x is a single parameter block, we only have one row of in the Jacobian

    // Compute derivatives
    Matrix<double,4,3> J_delta = dqpd_dd(q_nb.toEigen());
    Matrix<double,2,4> J_q = ddqt_dq(h.toEigen());

    // Compute minimal Jacobian of the state dynamics
    Matrix<double,4,3> J_autodiff_delta = J_autodiff * J_delta;
    Matrix<double,2,3> J = J_q * J_autodiff_delta;

    // Prints
    cout << "r = \n" << r << "\n\n";
    cout << "J_autodiff = \n" << J_autodiff << "\n\n";
    cout << "J_delta = \n" << J_delta << "\n\n";
    cout << "J_autodiff_delta = \n" << J_autodiff_delta << "\n\n";
    cout << "J = \n" << J << "\n\n";
    cout << "J_numerical = \n" << J_numerical << "\n\n";
  }
}

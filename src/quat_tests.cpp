#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "common_cpp/common.h"
#include "common_cpp/quaternion.h"
#include <chrono>
#include "quaternion_heap.h"
#include "quaternion_stack.h"

using namespace std;
using namespace Eigen;


struct S3Plus1
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
    return true;
  }
};

struct S3Plus2
{
  template<typename T>
  bool operator()(const T* q1, const T* delta, T* q2) const
  {
    Map<const Matrix<T,3,1>> delta_(delta);
    quat2::Quaternion2<const T> q1_(q1);
    quat2::Quaternion2<T> q2_(q2);
    q2_ = q1_ + Matrix<T,3,1>(delta_);
    return true;
  }
};

struct S3Plus3
{
  template<typename T>
  bool operator()(const T* q1, const T* delta, T* q2) const
  {
    Map<const Matrix<T,3,1>> delta_(delta);
    quat3::Quaternion3<const T> q1_(q1);
    quat3::Quaternion3<T> q2_(q2);
    q2_ = q1_ + Matrix<T,3,1>(delta_);
    return true;
  }
};


int main()
{
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  srand(seed);

  common::Quaternion<double> q_rand(Vector4d::Random().normalized());
  const Vector3d _delta = VectorXd::Random();
  quat3::Quaternion3<const double> _q1(q_rand.data());
  quat3::Quaternion3<double> _q2;

  // Test in Ceres
  ceres::AutoDiffLocalParameterization<S3Plus3,4,3> parameterization;
  parameterization.Plus(_q1.data(),_delta.data(),_q2.data());
  double _J[12];
  parameterization.ComputeJacobian(_q2.data(),_J);
  Map<Matrix<double,4,3>> J_(_J);

  cout << "q1 = \n" << _q1 << endl;
  cout << "\ndelta = \n" << _delta << endl;
  cout << "\nq2 = \n" << _q2 << endl;
  cout << "\nq2' = \n" << _q1 + _delta << endl;
  cout << "\nJ = \n" << J_ << "\n\n\n";

  // Profile each method
  int N = 1e6;
  auto t0 = std::chrono::system_clock::now();
  for (int i = 0; i < N; ++i)
  {
    common::Quaternion<double> q1(Vector4d::Random().normalized());
    const Vector3d delta = VectorXd::Random();
    common::Quaternion<double> q2;
    ceres::AutoDiffLocalParameterization<S3Plus1,4,3> parameterization;
    parameterization.Plus(q1.data(),delta.data(),q2.data());
    double J[12];
    parameterization.ComputeJacobian(q2.data(),J);
  }
  auto t1 = std::chrono::system_clock::now();
  for (int i = 0; i < N; ++i)
  {
    common::Quaternion<double> q1(Vector4d::Random().normalized());
    const Vector3d delta = VectorXd::Random();
    common::Quaternion<double> q2;
    ceres::AutoDiffLocalParameterization<S3Plus2,4,3> parameterization;
    parameterization.Plus(q1.data(),delta.data(),q2.data());
    double J[12];
    parameterization.ComputeJacobian(q2.data(),J);
  }
  auto t2 = std::chrono::system_clock::now();
  for (int i = 0; i < N; ++i)
  {
    common::Quaternion<double> q1(Vector4d::Random().normalized());
    const Vector3d delta = VectorXd::Random();
    common::Quaternion<double> q2;
    ceres::AutoDiffLocalParameterization<S3Plus3,4,3> parameterization;
    parameterization.Plus(q1.data(),delta.data(),q2.data());
    double J[12];
    parameterization.ComputeJacobian(q2.data(),J);
  }
  auto t3 = std::chrono::system_clock::now();
  auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto time3 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  cout << "Class 1 time: " << time1 << endl;
  cout << "Class 2 time: " << time2 << endl;
  cout << "Class 3 time: " << time3 << endl;

  return 0;
}

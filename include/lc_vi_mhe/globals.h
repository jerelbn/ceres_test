#pragma once
#include <Eigen/Eigen>
#include <set>
#include "geometry/quat.h"


namespace global
{

typedef Eigen::Matrix<double,9,1> Vec9;
typedef Eigen::Matrix<double,15,1> Vec15;
const Eigen::Vector3d e3(0, 0, 1);
const double g = 9.81; // gravity

struct Measurement
{
  enum { IMU, VO };
  int type;
  double t; // time stamp
  Eigen::Vector3d acc; // IMU
  Eigen::Vector3d gyro; // IMU
  Eigen::Vector3d p; // VO
  quat::Quatd q; // VO
  bool operator<(const Measurement& meas) const { return t < meas.t; }
};
typedef std::multiset<Measurement> MeasurementList;

} // namespace global

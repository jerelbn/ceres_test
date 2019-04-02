#pragma once
#include <Eigen/Eigen>
#include <random>
#include <fstream>
#include "geometry/quat.h"
#include "lc_vi_mhe/globals.h"

using namespace std;
using namespace Eigen;
using namespace global;

class TrueState
{
public:
  Vector3d p; // body w.r.t. inertial in inertial frame
  Vector3d v;
  Vector3d a;
  quat::Quatd q; // rotation inertial to body
  Vector3d omega; // angular rate in body frame
  Vector3d ba; // accel bias
  Vector3d bg; // gyro bias
  Vector3d p_bc; // body to camera translation in body frame
  quat::Quatd q_bc; // body to camera rotation
  normal_distribution<double> acc_dist; // accel noise distribution
  normal_distribution<double> gyro_dist; // gyro noise distribution
  normal_distribution<double> vot_dist; // VOT noise distribution
  normal_distribution<double> vor_dist; // VOR noise distribution
  bool noise_on; // add noise to accel and gyro measurements
  ofstream log_file;
  double imu_update_rate; // Hz
  double vo_update_rate; // Hz
  double imu_t_prev;
  double vo_t_prev;
  Vector3d pk; // keyframe position
  quat::Quatd qk; // keyframe attitude

  TrueState();
  ~TrueState();
  TrueState(const Vector3d& _ba, const Vector3d& _bg,
            const Vector3d &_p_bc, const quat::Quatd &_q_bc,
            const double& sigma_acc, const double& sigma_gyro, const bool& _noise_on,
            const double &sigma_vot, const double &sigma_vor,
            const double &_imu_update_rate, const double &_vo_update_rate);

  void update(const double& t);
  void getState(const double& t, Vector3d& p, Vector3d& v, Vector3d& a, quat::Quatd& q, Vector3d& omega);
  Vector3d getAccel(default_random_engine& rng);
  Vector3d getGyro(default_random_engine& rng);
  Vector3d getVOT(default_random_engine& rng);
  quat::Quatd getVOR(default_random_engine& rng);
  void getMeasurements(const double& t, MeasurementList& meas_list, default_random_engine &rng);
  Vec15 toVec();
  void log(const double& t);
};

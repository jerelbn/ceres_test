#include "lc_vi_mhe/true_state.h"

TrueState::TrueState()
{
  update(0);
  pk = p;
  qk = q;
  ba.setZero();
  bg.setZero();
  p_bc.setZero();
  acc_dist = normal_distribution<double>(0.0, 0.5);
  gyro_dist = normal_distribution<double>(0.0, 0.1);
  vot_dist = normal_distribution<double>(0.0, 0.05);
  vor_dist = normal_distribution<double>(0.0, 0.01);
  noise_on = false;
  imu_update_rate = 250;
  vo_update_rate = 10;
  imu_t_prev = 1e9;
  vo_t_prev = 1e9;
  log_file.open("../logs/true_state.bin");
  log(0);
}

TrueState::~TrueState()
{
  log_file.close();
}

TrueState::TrueState(const Vector3d& _ba, const Vector3d& _bg,
                     const Vector3d& _p_bc, const common::Quaterniond& _q_bc,
                     const double& sigma_acc, const double& sigma_gyro, const bool& _noise_on,
                     const double& sigma_vot, const double& sigma_vor,
                     const double& _imu_update_rate, const double& _vo_update_rate)
{
  update(0);
  pk = p;
  qk = q;
  ba = _ba;
  bg = _bg;
  p_bc = _p_bc;
  q_bc = _q_bc;
  acc_dist = normal_distribution<double>(0.0, sigma_acc);
  gyro_dist = normal_distribution<double>(0.0, sigma_gyro);
  vot_dist = normal_distribution<double>(0.0, sigma_vot);
  vor_dist = normal_distribution<double>(0.0, sigma_vor);
  noise_on = _noise_on;
  imu_update_rate = _imu_update_rate;
  vo_update_rate = _vo_update_rate;
  imu_t_prev = 1e9;
  vo_t_prev = 1e9;
  log_file.open("../logs/true_state.bin");
  log(0);
}

void TrueState::update(const double& t)
{
  getState(t, p, v, a, q, omega);
  log(t);
}

void TrueState::getState(const double &t, Vector3d &p, Vector3d &v, Vector3d &a, common::Quaterniond &q, Vector3d &omega)
{
  p(0) = sin(t);
  p(1) = sin(t);
  p(2) = sin(t);

  v(0) = cos(t);
  v(1) = cos(t);
  v(2) = cos(t);

  a(0) = -sin(t);
  a(1) = -sin(t);
  a(2) = -sin(t);

  double roll = sin(t);
  double pitch = sin(t);
  double yaw = sin(t);
  q = common::Quaterniond::fromEuler(roll, pitch, yaw);

  Matrix3d A;
  A.setZero();
  A(0,0) = 1.0;
  A(0,2) = -sin(pitch);
  A(1,1) = cos(roll);
  A(1,2) = sin(roll) * cos(pitch);
  A(2,1) = -sin(roll);
  A(2,2) = cos(roll) * cos(pitch);
  double roll_dot = cos(t);
  double pitch_dot = cos(t);
  double yaw_dot = cos(t);
  omega = A * Vector3d(roll_dot, pitch_dot, yaw_dot);
}

Vector3d TrueState::getAccel(default_random_engine& rng)
{
  Vector3d meas = q.rotp(a - g * e3) + ba;
  if (noise_on)
  {
    Vector3d noise(acc_dist(rng), acc_dist(rng), acc_dist(rng));
    return meas + noise;
  }
  else
    return meas;
}

Vector3d TrueState::getGyro(default_random_engine& rng)
{
  Vector3d meas = omega + bg;
  if (noise_on)
  {
    Vector3d noise(gyro_dist(rng), gyro_dist(rng), gyro_dist(rng));
    return meas + noise;
  }
  else
    return meas;
}

Vector3d TrueState::getVOT(default_random_engine &rng)
{
  if (noise_on)
  {
    Vector3d noise(vor_dist(rng), vor_dist(rng), vor_dist(rng));
    Vector3d p_ck = q_bc.rotp(q.rotp(pk + qk.rota(p_bc) - (p + q.rota(p_bc)))) + noise;
    return p_ck.normalized();
  }
  else
  {
    Vector3d p_ck = q_bc.rotp(q.rotp(pk + qk.rota(p_bc) - (p + q.rota(p_bc))));
    return p_ck.normalized();
  }
}

common::Quaterniond TrueState::getVOR(default_random_engine &rng)
{
  if (noise_on)
  {
    common::Quaterniond q_noise = common::Quaterniond::fromEuler(vor_dist(rng), vor_dist(rng), vor_dist(rng));
    return q_bc.inverse() * q.inverse() * qk * q_bc * q_noise;
  }
  else
    return q_bc.inverse() * q.inverse() * qk * q_bc;
}

void TrueState::getMeasurements(const double& t, MeasurementList &meas_list, default_random_engine& rng)
{
  if (fabs(t - imu_t_prev) >= 1.0 / imu_update_rate)
  {
    Measurement meas;
    meas.type = Measurement::IMU;
    meas.t = t;
    meas.acc = getAccel(rng);
    meas.gyro = getGyro(rng);
    imu_t_prev = round(10000 * t) / 10000;
    meas_list.insert(meas);
  }
  if (fabs(t - vo_t_prev) >= 1.0 / vo_update_rate)
  {
    Measurement meas;
    meas.type = Measurement::VO;
    meas.t = t;
    meas.p = getVOT(rng);
    meas.q = getVOR(rng);
    vo_t_prev = round(10000 * t) / 10000;
    meas_list.insert(meas);
    pk = p;
    qk = q;
  }
}

Vec15 TrueState::toVec()
{
  Vec15 vec;
  vec.segment<3>(0) = p;
  vec.segment<3>(3) = v;
  vec.segment<3>(6) = a;
  vec.segment<3>(9) = q.eulerVector();
  vec.segment<3>(12) = omega;
  return vec;
}

void TrueState::log(const double &t)
{
  Vec15 vec = toVec();
  log_file.write((char*)&t, sizeof(double));
  log_file.write((char*)vec.data(), sizeof(double) * 15);
}

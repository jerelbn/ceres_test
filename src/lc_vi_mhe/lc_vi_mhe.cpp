#include "lc_vi_mhe/lc_vi_mhe.h"


using namespace std;
using namespace Eigen;


MHE::MHE()
{
  ba_.setZero();
  bg_.setZero();
}


MHE::MHE(const MeasurementList& meas_list)
{
  separateMeasurements(meas_list);
  ba_.setZero();
  bg_.setZero();
}


void MHE::separateMeasurements(const MeasurementList& meas_list)
{
  // Separate IMU from VO measurements
  imu_list_.clear();
  vo_list_.clear();
  for (auto& m : meas_list)
  {
    if (m.type == Measurement::IMU)
      imu_list_.push_back(m);
    else if (m.type == Measurement::VO)
      vo_list_.push_back(m);
    else
      throw runtime_error("MH broken at measurement separation!");
  }

  // Determine number of nodes
  N_ = vo_list_.size();
  x_.resize(7,N_);
  v_.resize(3,N_);
}


void MHE::buildGraph()
{
  // Create a new ceres problem
  if (problem_ != nullptr)
    delete problem_;
  problem_ = new ceres::Problem;

  // Initialize parameters
  // set first pose and velocity to identity and zero
  // integrate imu to get following poses and velocities
  x_.col(0) = common::Transformd().toEigen();
  v_.col(0).setZero();
}


void MHE::preintegrationTest(const double& dt, TrueState& truth)
{
  // Test pre-integration between all keyframes
  Vector3d alpha, beta;
  common::Quaterniond gamma;
  Vector3d pk, pk2, vk, vk2;
  common::Quaterniond qk, qk2;
  Vector3d dummy1, dummy2;
  double tk, tk2;

  // Integrate IMU between keyframes and check factor error
  bool midpoint_integration = false;
  int vo_idx = 0;
  Measurement vo_meas = vo_list_[vo_idx];
  tk2 = vo_meas.t;
  truth.getState(tk2, pk2, vk2, dummy1, qk2, dummy2);
  for (auto& imu : imu_list_)
  {
    if (tk2 <= imu.t)
    {
      if (vo_idx > 0)
      {
        // Estimates
        double dtk = tk2 - tk;
        static common::Quaterniond qbg = common::Quaterniond::exp(-truth.bg * dtk);
        Vector3d alpha_hat = qk.rotp(pk2 - pk - vk * dtk - 0.5 * g * global::e3 * dtk *dtk);
        Vector3d beta_hat = qk.rotp(vk2 - vk - g * global::e3 * dtk);
        common::Quaterniond gamma_hat = qk.inverse() * qk2 * qbg.inverse();

        // Errors
        Vector3d alpha_error = alpha - alpha_hat;
        Vector3d beta_error = beta - beta_hat;
        Vector3d gamma_error = gamma - gamma_hat;

        cout << "\nk = " << vo_idx << endl;
        cout << "alpha_error = " << alpha_error.transpose() << ", norm = " << alpha_error.norm() << endl;
        cout << "beta_error = " << beta_error.transpose() << ", norm = " << beta_error.norm() << endl;
        cout << "gamma_error = " << gamma_error.transpose() << ", norm = " << gamma_error.norm() << endl;
      }
      tk = tk2;
      pk = pk2;
      vk = vk2;
      qk = qk2;
      alpha.setZero();
      beta.setZero();
      gamma = common::Quaterniond();
      ++vo_idx;
      vo_meas = vo_list_[vo_idx];
      tk2 = vo_meas.t;
      truth.getState(tk2, pk2, vk2, dummy1, qk2, dummy2);
    }

    if (!midpoint_integration)
    {
      // Euler integration
      alpha += beta * dt;
      beta += gamma.rota(imu.acc - truth.ba) * dt;
      gamma += (imu.gyro) * dt;
    }
    else
    {
      // Midpoint integration on alpha
      Vector3d beta_mid = beta + gamma.rota(imu.acc) * dt/2;
      alpha += beta_mid * dt;
      beta += gamma.rota(imu.acc) * dt;
      gamma += (imu.gyro) * dt;
    }
  }
}

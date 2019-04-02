// This is a simple test for one window of a loosely-coupled visual-inertial moving horizon estimator
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <random>
#include <experimental/filesystem>
#include "geometry/quat.h"
#include "lc_vi_mhe/globals.h"
#include "lc_vi_mhe/true_state.h"
#include "lc_vi_mhe/lc_vi_mhe.h"


using namespace std;
using namespace Eigen;



int main()
{
  // Create log folder
  if(!experimental::filesystem::exists("../logs"))
      if (experimental::filesystem::create_directory("../logs"))
        cout << "*** Created directory: ../logs/ ***\n";

  // Random number generator
//  default_random_engine rng((unsigned)time(nullptr));
  default_random_engine rng(0);

  // Initialize truth data and measurement data
  double imu_update_rate = 250;
  double vo_update_rate = 2;

  double t = 0;
  double dt = round(1.0 / imu_update_rate * 1e6) / 1e6;
  double tf = 1.0;

  Vector3d ba(0.1, 0.2, 0.3);
  Vector3d bg(0.1, 0.2, 0.3);
  Vector3d p_bc(0, 0, 0);
  quat::Quatd q_bc;

  TrueState truth(ba, // accel bias
                  bg, // gyro bias
                  p_bc, // body to camera translation
                  q_bc, // body to camera rotation
                  0.5, // accel noise stdev
                  0.1, // gyro noise stdev
                  false, // IMU noise added to measurements
                  0.05, // VOT noise stdev
                  0.01, // VOR noise stdev
                  imu_update_rate, // IMU update rate
                  vo_update_rate // VO update rate
                  );

  MeasurementList meas;
  truth.getMeasurements(t, meas, rng);

  // Calculate truth and collect measurements
  while (fabs(t - tf) > 1e-6)
  {
    t += dt;
    truth.update(t);
    truth.getMeasurements(t, meas, rng);
  }

  // Pass measurements into estimator for optimization
  MHE mhe(meas);
  mhe.preintegrationTest(dt, truth);

  return 0;
}

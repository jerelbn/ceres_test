#pragma once
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <set>
#include "common_cpp/quaternion.h"
#include "common_cpp/transform.h"
#include "lc_vi_mhe/globals.h"
#include "lc_vi_mhe/true_state.h"


using namespace std;
using namespace Eigen;
using namespace global;


class MHE
{
public:
  int N_; // number of nodes
  vector<Measurement> imu_list_, vo_list_;
  vector<vector<Measurement>> imu_lists_;
  MatrixXd x_; // 7xN_ matrix of poses (pos/quat)
  MatrixXd v_; // 3xN_ matrix of velocities
  Vector3d ba_, bg_;
  ceres::Problem* problem_;

  MHE();
  MHE(const MeasurementList& meas_list);

  void separateMeasurements(const MeasurementList& meas_list);
  void buildGraph();
  void preintegrationTest(const double& dt, TrueState &truth);

  // collect measurements
  // split measurements into IMU and VO lists
  // initialize parameters for graph
  // add parameters to Ceres
  // add residuals
  // solve
};

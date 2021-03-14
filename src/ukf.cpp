#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_) {
    previous_timestamp_ = meas_package.timestamp_;

    Initialise(meas_package);
    is_initialized_ = true;
  } else {
    float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(dt);
  }

}

void UKF::Initialise(MeasurementPackage meas_package) {
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    x_ << meas_package.raw_measurements_[0], 
          meas_package.raw_measurements_[1], 
          0, 
          0,
          0;
  }
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    // computation of v is taken from 
    // https://knowledge.udacity.com/questions/398805

    double rho = meas_package.raw_measurements_(0);
    double phi = meas_package.raw_measurements_(1);
    double rho_d = meas_package.raw_measurements_(2);
    double p_x = rho * cos(phi);
    double p_y = rho * sin(phi);
    
    double vx = rho_d * cos(phi);
    double vy = rho_d * sin(phi);
    double v = sqrt(vx * vx + vy * vy);
    
    x_ << p_x, 
          p_y, 
          v,
          0, 
          0;
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  MatrixXd Xsig;
  GenerateSigmaPoints(&Xsig);

  MatrixXd Xsig_aug;
  AugmentedSigmaPoints(&Xsig, &Xsig_aug);

  SigmaPointPrediction(&Xsig_aug, delta_t);

  PredictMeanAndCovariance();
}


void UKF::GenerateSigmaPoints(MatrixXd* Xsig) {
  // create sigma point matrix
  MatrixXd Xsig_temp = MatrixXd(n_x_, 2 * n_x_ + 1);

  // calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  Xsig_temp.col(0) = x_;
  
  MatrixXd first = sqrt(3)*A;
  for(int i = 0; i<n_x_; i++) {
    Xsig_temp.col(1 + i) = first.col(i)+x_;
  }
  
  MatrixXd second = -sqrt(3)*A;
  for(int i = 0; i<n_x_; i++) {
    Xsig_temp.col(n_x_ + 1 + i) = second.col(i)+x_;
  }
  *Xsig = Xsig_temp;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig, MatrixXd* Xsig_aug) {
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug_temp = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.head(n_x_) = x_;
  
  P_aug.topLeftCorner(n_x_, n_x_) = P_;

  P_aug(n_aug_-1, n_aug_-1) = pow(std_yawdd_, 2);
  P_aug(n_aug_-2, n_aug_-2) = pow(std_a_, 2);

  Xsig_aug_temp.col(0) = x_aug;

  MatrixXd A = P_aug.llt().matrixL();

  MatrixXd first = sqrt(5) * A;
  for(int i = 0; i<n_aug_; i++) {
    Xsig_aug_temp.col(1 + i) = first.col(i)+x_aug;
  }
  
  MatrixXd second = -sqrt(5)*A;
  for(int i = 0; i<n_aug_; i++) {
    Xsig_aug_temp.col(n_aug_ + 1 + i) = second.col(i)+x_aug;
  }

  *Xsig_aug = Xsig_aug_temp;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_aug, double delta_t) {
  VectorXd x_old = VectorXd(n_x_);

  for(int i=0; i<2*n_aug_+1; i++) {
    VectorXd col = Xsig_aug->col(i);
    x_old = col.head(n_x_);
    
    float p_x = col[0];
    float p_y = col[1];
    float v = col[2];
    float psi = col[3];
    float psi_dot = col[4];
    float nue_a = col[5];
    float nue_psi = col[6];

    float x_1_delta;
    float x_2_delta;

    if (psi_dot != 0) {
      x_1_delta = v / psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)) + 0.5 * delta_t * delta_t * cos(psi) * nue_a;
      x_2_delta = v / psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)) + 0.5 * delta_t * delta_t * sin(psi) * nue_a;
    } else {
      x_1_delta = v * cos(psi) * delta_t + 0.5 * delta_t * delta_t * cos(psi) * nue_a;
      x_2_delta = v * sin(psi) * delta_t + 0.5 * delta_t * delta_t * sin(psi) * nue_a;
    }
    float x_3_delta = delta_t * nue_a;
    float x_4_delta = psi_dot * delta_t + 0.5 * delta_t * delta_t * nue_psi;
    float x_5_delta = delta_t * nue_psi;

    VectorXd x_delta = VectorXd(n_x_);

    x_delta << x_1_delta, x_2_delta, x_3_delta, x_4_delta, x_5_delta;

    VectorXd x_new = x_old + x_delta;

    Xsig_pred_.col(i) = x_new;
  }

}

void UKF::PredictMeanAndCovariance() {

  // create vector for weights
  VectorXd weights = VectorXd(2*n_aug_+1);
  
  // set weights
  weights[0] = 1.0 * lambda_ / (lambda_ + n_aug_);
  for (int i=1; i< 2*n_aug_+1; i++) {
    weights[i] = 1.0 / (2 * (lambda_ + 7));
  }

  // predict state mean
  // x = Xsig_pred * weights;
  for (int i=0; i<Xsig_pred_.rows(); i++) {
    x_[i] = Xsig_pred_.row(i) * weights;
  }

  // predict state covariance matrix
  MatrixXd x_adjusted = MatrixXd(Xsig_pred_.rows(), Xsig_pred_.cols());
  for (int i=0; i<Xsig_pred_.cols(); i++) {
    x_adjusted.col(i) = Xsig_pred_.col(i) - x_;
  }

  MatrixXd weighted_x_adjusted = MatrixXd(Xsig_pred_.rows(), Xsig_pred_.cols());
  for (int i=0; i<Xsig_pred_.rows(); i++) {
    VectorXd a = Xsig_pred_.row(i).array() * weights.transpose().array();
    weighted_x_adjusted.row(i) = a;
  }

  P_ = weighted_x_adjusted * x_adjusted.transpose();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
}
#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
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
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
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
  
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  weights_ = VectorXd(2*n_aug_+1);
  
  weights_[0] = 1.0 * lambda_ / (lambda_ + n_aug_);
  for (int i=1; i< 2*n_aug_+1; i++) {
    weights_[i] = 1.0 / (2 * (lambda_ + n_aug_));
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    previous_timestamp_ = meas_package.timestamp_;

    Initialise(meas_package);
    is_initialized_ = true;
    return;
  }

  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  }

  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
}

void UKF::Initialise(MeasurementPackage meas_package) {
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
  x_.fill(0.0);

  if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
    x_ << meas_package.raw_measurements_[0], 
          meas_package.raw_measurements_[1], 
          0, 
          0,
          0;

  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
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
  MatrixXd Xsig_temp = MatrixXd::Zero(n_x_, 2 * n_x_ + 1);

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
  VectorXd x_aug = VectorXd::Zero(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug_temp = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.head(n_x_) = x_;
  
  P_aug.topLeftCorner(n_x_, n_x_) = P_;

  P_aug(n_aug_-1, n_aug_-1) = pow(std_yawdd_, 2);
  P_aug(n_aug_-2, n_aug_-2) = pow(std_a_, 2);

  Xsig_aug_temp.col(0) = x_aug;

  MatrixXd A = P_aug.llt().matrixL();

  MatrixXd first = sqrt(lambda_ + n_aug_) * A;
  for(int i = 0; i<n_aug_; i++) {
    Xsig_aug_temp.col(1 + i) = first.col(i)+x_aug;
  }
  
  MatrixXd second = -sqrt(lambda_ + n_aug_)*A;
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
  // predict state mean
  // x = Xsig_pred * weights;
  for (int i=0; i<Xsig_pred_.rows(); i++) {
    x_[i] = Xsig_pred_.row(i) * weights_;
  }

  // predict state covariance matrix
  MatrixXd x_adjusted = MatrixXd::Zero(Xsig_pred_.rows(), Xsig_pred_.cols());

  for (int i=0; i<Xsig_pred_.cols(); i++) {
    x_adjusted.col(i) = Xsig_pred_.col(i) - x_;
  }

  MatrixXd weighted_x_adjusted = MatrixXd::Zero(Xsig_pred_.rows(), Xsig_pred_.cols());

  for (int i=0; i<Xsig_pred_.rows(); i++) {
    VectorXd a = Xsig_pred_.row(i).array() * weights_.transpose().array();
    weighted_x_adjusted.row(i) = a;
  }

  P_ = weighted_x_adjusted * x_adjusted.transpose();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z_pred;
  MatrixXd S;
  MatrixXd Zsig;

  PredictLidarMeasurement(&Zsig, &z_pred, &S);

  VectorXd z = meas_package.raw_measurements_;
  UpdateState(&z, &Zsig, &z_pred, &S, 2);
}

void UKF::PredictLidarMeasurement(MatrixXd* Zsig, VectorXd* z_pred, MatrixXd* S) {
  // create matrix for sigma points in measurement space
  int n_z = 2;
  MatrixXd Zsig_temp = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred_temp = VectorXd::Zero(n_z);

  // transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); i++) {
      VectorXd c = Xsig_pred_.col(i);
      float px = c[0];
      float py = c[1];

      VectorXd z = VectorXd(n_z);
      z << px, py;
      Zsig_temp.col(i) = z;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred_temp = z_pred_temp + (weights_(i) * Zsig_temp.col(i));
  }

  *Zsig = Zsig_temp;
  *z_pred = z_pred_temp;

  MatrixXd Z_adjusted = MatrixXd::Zero(Zsig_temp.rows(), Zsig_temp.cols());

  for (int i=0; i<Zsig_temp.cols(); i++) {
    Z_adjusted.col(i) = Zsig_temp.col(i) - z_pred_temp;
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << \
    pow(std_laspx_, 2), 0, \
    0,                  pow(std_laspy_, 2);

  MatrixXd weighted_Z_adjusted = MatrixXd::Zero(Z_adjusted.rows(), Z_adjusted.cols());

  for (int i=0; i<Z_adjusted.rows(); i++) {
    VectorXd a = Z_adjusted.row(i).array() * weights_.transpose().array();
    weighted_Z_adjusted.row(i) = a;
  }

  *S = weighted_Z_adjusted * Z_adjusted.transpose() + R;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  VectorXd z_pred;
  MatrixXd S;
  MatrixXd Zsig;

  PredictRadarMeasurement(&Zsig, &z_pred, &S);

  VectorXd z = meas_package.raw_measurements_;
  UpdateState(&z, &Zsig, &z_pred, &S, 3);
}

void UKF::PredictRadarMeasurement(MatrixXd* Zsig, VectorXd* z_pred, MatrixXd* S) {
  // set measurement dimension, radar can measure r, phi, and r_dot

  // create matrix for sigma points in measurement space
  int n_z = 3;
  MatrixXd Zsig_temp = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred_temp = VectorXd::Zero(n_z);

  // transform sigma points into measurement space
  for (int i=0; i<Xsig_pred_.cols(); i++) {
      VectorXd c = Xsig_pred_.col(i);
      float px = c[0];
      float py = c[1];
      float v = c[2];
      float psi = c[3];
      float psi_dot = c[4];
    
      float measurement_rho = sqrt(pow(px, 2) + pow(py, 2));
      float measurement_psi = atan2(py, px);
      float measurement_rho_dot = (px * cos(psi) * v + py * sin(psi) * v) / (sqrt(pow(px, 2) + pow(py, 2)));

      VectorXd z = VectorXd(3);
      z << measurement_rho, measurement_psi, measurement_rho_dot;
      Zsig_temp.col(i) = z;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred_temp = z_pred_temp + (weights_(i) * Zsig_temp.col(i));
  }

  *Zsig = Zsig_temp;
  *z_pred = z_pred_temp;

  MatrixXd R = MatrixXd(n_z, n_z);
  R << \
    pow(std_radr_, 2), 0,                   0, \
    0,                 pow(std_radphi_, 2), 0, \
    0,                 0,                   pow(std_radrd_, 2);

  MatrixXd Z_adjusted = MatrixXd::Zero(Zsig_temp.rows(), Zsig_temp.cols());

  for (int i=0; i<Zsig_temp.cols(); i++) {
    Z_adjusted.col(i) = Zsig_temp.col(i) - z_pred_temp;
  }

  MatrixXd weighted_Z_adjusted = MatrixXd::Zero(Z_adjusted.rows(), Z_adjusted.cols());

  for (int i=0; i<Z_adjusted.rows(); i++) {
    VectorXd a = Z_adjusted.row(i).array() * weights_.transpose().array();
    weighted_Z_adjusted.row(i) = a;
  }

  *S = weighted_Z_adjusted * Z_adjusted.transpose() + R;
}

void UKF::UpdateState(VectorXd* z, MatrixXd* Zsig, VectorXd* z_pred, MatrixXd* S, int n_z) {
  // Code copied from the solution in the class

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);


  MatrixXd x_adjusted = MatrixXd(Xsig_pred_.rows(), Xsig_pred_.cols());
  MatrixXd z_adjusted = MatrixXd(Zsig->rows(), Zsig->cols());
  for (int i=0; i<2 * n_aug_ + 1; i++) {
    VectorXd x_adjusted = Xsig_pred_.col(i) - x_;
    while (x_adjusted(1)> M_PI) x_adjusted(1)-=2.*M_PI;
    while (x_adjusted(1)<-M_PI) x_adjusted(1)+=2.*M_PI;
    
    VectorXd z_adjusted = Zsig->col(i) - *z_pred;
    while (z_adjusted(1)> M_PI) z_adjusted(1)-=2.*M_PI;
    while (z_adjusted(1)<-M_PI) z_adjusted(1)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_adjusted * z_adjusted.transpose();
  }

  VectorXd z_diff = *z - *z_pred;
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // calculate Kalman gain K;
  MatrixXd K = Tc * S->inverse();

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff; 
  P_ = P_ - K * *S * K.transpose();
}

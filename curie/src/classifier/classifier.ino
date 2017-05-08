#include "CurieIMU.h"
#include "network.hpp"
#include "network-io.hpp"

Network *network;

void setup() {
  /* Initialise Serial communication */
  Serial.begin(9600);
  while(!Serial) ;    // wait for serial port to connect.

  /* Initialise the IMU */
  CurieIMU.begin();

  /* Calibrate the IMU's accelerometer */
  CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);

  /* Initialise the network */
  network =  new Network(20, 10, 1, 0.3, 0.9, 0.5);
}

void loop() {}

/*
 *  Program to record the IMU data and send it to 
 *  the computer for processing & training of the neural nework
 * 
 *  For use with someone to supervise the exerciser and manually mark the start and end of exercises
 */

#include "CurieIMU.h"

bool moving = false;                
bool calibrateOffsets = true;

unsigned long readingInterval = 100;  // Time between readings when logging

int ax, ay, az;         // Accelerometer values
int acceleration;       // Overall acceleration magnitude

void setup() {
  Serial.begin(9600); // initialize Serial communication
  while(!Serial) ;    // wait for serial port to connect.

  /* Initialise the IMU */
  CurieIMU.begin();

  /* Calibrate the IMU's gyrometer */
  CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);
}

void loop() {
  CurieIMU.readAccelerometer(ax, ay, az);
  acceleration = abs(ax) + abs(ay) + abs(az);
  Serial.println(acceleration);
  delay(readingInterval);
}


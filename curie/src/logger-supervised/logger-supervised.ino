/*
 *  Program to record the IMU data and send it to 
 *  the computer for processing & training of the neural nework
 * 
 *  For use with someone to supervise the exerciser and manually mark the start and end of exercises
 */

#include "CurieIMU.h"

bool moving = false;                
bool calibrateOffsets = true;

unsigned long readingInterval = 75;  // Time between readings when logging

int ax, ay, az;         // Accelerometer values

void setup() {
  /* Initialise the IMU */
  CurieIMU.begin();

  /* Calibrate the IMU's gyrometer */
  CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);

  /* Initialise the builtin LED to blink and indicate activity */
  pinMode(LED_BUILTIN, OUTPUT);
  /* Leave LED on while waiting for Serial*/
  digitalWrite(LED_BUILTIN, HIGH);
  
  Serial.begin(9600); // initialize Serial communication
  while(!Serial) ;    // wait for serial port to connect.
}

void loop() {
  /* Blink the LED while connected*/
  digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  
  CurieIMU.readAccelerometer(ax, ay, az);
  Serial.print(ax); Serial.print(" ");
  Serial.print(ay); Serial.print(" ");
  Serial.println(az); Serial.print(" ");

  delay(readingInterval);
}


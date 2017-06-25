/*
 *  Program to detect exercises, record the IMU data and send it to 
 *  the computer for processing & training of the neural nework
 * 
 *  For use when someone to supervise the exerciser cannot be found.
 * 
    Motion detection code taken from the 'MotionDetect' Curie example,
    Copyright (c) 2016 Intel Corporation.  All rights reserved.
    See the bottom of this file for license terms.
*/

#include "CurieIMU.h"

bool moving = false;                
bool calibrateOffsets = true;

unsigned long cooldownTime = 750;     //Cooldown period before another switch can happen, in milliseconds
unsigned long lastSwitchTime = 0;     // Time of the last switch in 'moving' state
unsigned long interruptTime = 0;      // Time of the last interrupt
unsigned long readingInterval = 75;  // Time between readings when logging, in milliseconds

int ax, ay, az;         // Accelerometer values

void setup() {
  Serial.begin(9600); // initialize Serial communication
  while(!Serial) ;    // wait for serial port to connect.

  /* Initialise the IMU */
  CurieIMU.begin();
  CurieIMU.attachInterrupt(eventCallback);

  /* Calibrate the IMU's gyrometer */
  CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);
  
  /* Enable Motion Detection */
  CurieIMU.setDetectionThreshold(CURIE_IMU_MOTION, 20); // mg
  CurieIMU.setDetectionDuration(CURIE_IMU_MOTION, 10);  // number of consecutive positive samples required to detect motion
  CurieIMU.interrupts(CURIE_IMU_MOTION);

  /* Enable Zero Motion Detection */
  CurieIMU.setDetectionThreshold(CURIE_IMU_ZERO_MOTION, 35 );  // mg
  CurieIMU.setDetectionDuration(CURIE_IMU_ZERO_MOTION, 0.5);   // seconds
  CurieIMU.interrupts(CURIE_IMU_ZERO_MOTION);
}

void loop() {
  if (moving) {
    CurieIMU.readAccelerometer(ax, ay, az);
    Serial.print(ax); Serial.print(" ");
    Serial.print(ay); Serial.print(" ");
    Serial.println(az); Serial.print(" ");

    delay(readingInterval);
  } 
  else {
    // Do nothing
  }
}


static void eventCallback(void){
  interruptTime = millis();
  
  if (CurieIMU.getInterruptStatus(CURIE_IMU_MOTION) && !moving && (interruptTime - lastSwitchTime > cooldownTime)) {
    Serial.print("Motion detected after  ");
    Serial.print(interruptTime - lastSwitchTime);
    Serial.println("  milliseconds. Logging...");
    moving = true;
    lastSwitchTime = interruptTime;
  } 
  
  if (CurieIMU.getInterruptStatus(CURIE_IMU_ZERO_MOTION) && moving && (interruptTime - lastSwitchTime > cooldownTime)) {
    Serial.print("Motion ended after  ");
    Serial.print(interruptTime - lastSwitchTime);
    Serial.println("  milliseconds. Logging...");    
    moving = false;
    lastSwitchTime = interruptTime;
  } 

}

/*
 * Intel licence for 'MotionDetect' code.
 * 
   Copyright (c) 2016 Intel Corporation.  All rights reserved.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

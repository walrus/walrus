/*
 *  Program to detect exercises, record the IMU data and send it to 
 *  the computer for processing & training of the neural nework
 * 
    Motion detection code taken from the 'MotionDetect' Curie example,
    Copyright (c) 2016 Intel Corporation.  All rights reserved.
    See the bottom of this file for license terms.
*/

#include "CurieIMU.h"

bool moving = false;                

int ax, ay, az;         // accelerometer values
int gx, gy, gz;         // gyrometer values

void setup() {
  Serial.begin(9600); // initialize Serial communication
  while(!Serial) ;    // wait for serial port to connect.

  /* Initialise the IMU */
  CurieIMU.begin();
  CurieIMU.attachInterrupt(eventCallback);

  /* Enable Motion Detection */
  CurieIMU.setDetectionThreshold(CURIE_IMU_MOTION, 25); // mg
  CurieIMU.setDetectionDuration(CURIE_IMU_MOTION, 6);  // number of consecutive positive samples required to detect motion
  CurieIMU.interrupts(CURIE_IMU_MOTION);

  /* Enable Zero Motion Detection */
  CurieIMU.setDetectionThreshold(CURIE_IMU_ZERO_MOTION, 35);  // mg
  CurieIMU.setDetectionDuration(CURIE_IMU_ZERO_MOTION, 0.75);    // seconds
  CurieIMU.interrupts(CURIE_IMU_ZERO_MOTION);

  Serial.println("IMU initialisation complete, waiting for events...");
}

void loop() {

}


static void eventCallback(void){
  if (CurieIMU.getInterruptStatus(CURIE_IMU_MOTION) && !moving) {
    Serial.println("Motion detected; logging movement");
    moving = true;
  } 
  if (CurieIMU.getInterruptStatus(CURIE_IMU_ZERO_MOTION) && moving) {
    Serial.println("Motion ended; logging movement");
    moving = false;
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

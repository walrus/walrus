#include <ArduinoSTL.h>

#include "CurieIMU.h"
#include "network.hpp"

int numInputNodes;
int numHiddenNodes;
int numOutputNodes;
float learningRate;
float momentum;
float initialWeightMax;

int numHiddenWeights;
int numOutputWeights;

Network *network;

String config = "";

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

  /* Wait for data to start being sent */
  while (Serial.available() == 0) {
    Serial.println("Ready");
  }
  
  /* Read the network configuration over Serial */
  numInputNodes = Serial.parseInt();
  numHiddenNodes = Serial.parseInt();
  numOutputNodes = Serial.parseInt();

  numHiddenWeights = (numInputNodes + 1) * numHiddenNodes;
  numOutputWeights = (numHiddenNodes + 1) * numOutputNodes;

  learningRate = Serial.parseFloat();
  momentum = Serial.parseFloat();
  initialWeightMax = Serial.parseFloat();
  
  while (Serial.peek() != '#') {
    if (Serial.available() > 0) {
      config += Serial.read();
    }
  }

  while (Serial.available() == 0) {
    Serial.println("Received");
  }
  
  Serial.print("numInputNodes = ");
  Serial.println(numInputNodes);
  Serial.print("numHiddenNodes = ");
  Serial.println(numHiddenNodes);
  Serial.print("numOutputNodes = ");
  Serial.println(numOutputNodes);

  Serial.print("learningRate = ");
  Serial.println(learningRate);
  Serial.print("momentum = ");
  Serial.println(momentum);
  Serial.print("initialWeightMax = ");
  Serial.println(initialWeightMax);

  Serial.println("Finished");
}

void loop() {}

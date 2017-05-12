#include <ArduinoSTL.h>
#include <avr/pgmspace.h>

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
vector<vector<float>> hiddenWeights;
vector<vector<float>> outputWeights;

String config = "";

void setup() {
  /* Initialise Serial communication */
  Serial.begin(115200);
  while(!Serial) ;    // wait for serial port to connect.

  /* Initialise the IMU */
  CurieIMU.begin();

  /* Calibrate the IMU's accelerometer */
  CurieIMU.autoCalibrateAccelerometerOffset(X_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Y_AXIS, 0);
  CurieIMU.autoCalibrateAccelerometerOffset(Z_AXIS, 1);

  /* Wait for data to start being sent */
  while (!Serial.available()) {
    Serial.println(F("Ready"));
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

  /* Clear the input buffer */
  while (Serial.available() > 0) {
    Serial.read();
  }

  /* Send feedback over Serial for checking */

  Serial.print(F("Free memory: "));
  Serial.println(freeMemory());

  /* Notify computer that the data has been received */
  while (!Serial.available()) {
    Serial.println(F("Received"));
  }
  
  Serial.print(F("numInputNodes = "));
  Serial.println(numInputNodes);
  Serial.print(F("numHiddenNodes = "));
  Serial.println(numHiddenNodes);
  Serial.print(F("numOutputNodes = "));
  Serial.println(numOutputNodes);

  Serial.print(F("learningRate = "));
  Serial.println(learningRate);
  Serial.print(F("momentum = "));
  Serial.println(momentum);
  Serial.print(F("initialWeightMax = "));
  Serial.println(initialWeightMax);

  /* Init network */
  network = new Network(numInputNodes,
                     numHiddenNodes,
                     numOutputNodes,
                     learningRate,
                     momentum,
                     initialWeightMax);
  
  Serial.print(F("numHiddenWeights = "));
  Serial.println(numHiddenWeights);
  Serial.print(F("numOutputWeights = "));
  Serial.println(numOutputWeights);

  /* notify the computer that setup is done */
  while (!Serial.available()) {
    Serial.println(F("Finished setup"));
  }
}

void loop() {
  Serial.println(F("Running..."));
  Serial.print(F("Free memory: "));
  Serial.println(freeMemory());
  delay(1000);
}

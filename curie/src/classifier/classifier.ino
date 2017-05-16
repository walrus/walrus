#include <ArduinoSTL.h>

#include "network.hpp"

int numInputNodes = 20;
int numHiddenNodes = 10;
int numOutputNodes = 1;
float learningRate = 0.3;
float momentum = 0.9;
float initialWeightMax = 0.5;

Network *network;

void setup() {
  /* Initialise Serial communication */
  Serial.begin(115200);
  while(!Serial) ;

  /* Wait for script to notify readiness */
  while (!Serial.available()) {
    Serial.println(F("Ready"));
  }

  delay(1000);
  
  Serial.print(F("Free memory before test array reserved: "));
  Serial.println(freeMemory());

  float test[100] PROGMEM;

  Serial.print(F("Free memory after test vector reserved: "));
  Serial.println(freeMemory());

  std::vector<float> test2 PROGMEM;
  test2.resize(100) PROGMEM;

  Serial.print(F("Free memory after test2 vector resized: "));
  Serial.println(freeMemory());

  
  /* Init network */
  network = new Network(numInputNodes,
                     numHiddenNodes,
                     numOutputNodes,
                     learningRate,
                     momentum,
                     initialWeightMax) PROGMEM;
}

void loop() {
  Serial.println(F("Running..."));
  Serial.print(F("Free memory: "));
  Serial.println(freeMemory());
  delay(1000);
}

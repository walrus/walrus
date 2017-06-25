#ifndef NETWORK_A_H
#define NETWORK_A_H

#include <vector>
#include <random>

#include "arduino_config.h"

class Network_A {
private:
    float accumulatedInput;                     // AKA 'Accum' in the original code

    float hiddenNodes[numHiddenNodes];        // AKA 'Hidden' in the original code
    float outputNodes[numOutputNodes];        // AKA 'Output' in the original code

    void computeHiddenLayerActivations(float inputs[]);
    void computeOutputLayerActivations();

public:
    Network_A();
    std::string writeReport();
    float * classify(float inputs[]);

    int getNumInputNodes() const;
    int getNumHiddenNodes() const;
    int getNumOutputNodes() const;
    float getLearningRate() const;
    float getMomentum() const;
    float getInitialWeightMax() const;
    float getAccumulatedInput() const;
    const float * getHiddenNodes() const;
    const float * getOutputNodes() const;
};

#endif // NETWORK_A_H
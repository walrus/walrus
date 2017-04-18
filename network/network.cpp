/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 */



#include <string>

#include "header.h"

Network::Network(int InputNodes,
                 int HiddenNodes,
                 int OutputNodes,
                 float LearningRate,
                 float Momentum,
                 float InitialWeightMax,
                 float Success) {

}


void Network::initialiseHiddenWeights() {}


void Network::initialiseOutputWeights() {}


void Network::trainNetwork(float[InputNodes] inputs) {}


void Network::computeHiddenLayerActivations() {}


void Network::computeOutputLayerActivations() {}


void Network::backpropogateErrors() {}


void Network::updateInnerToHiddenWeights() {}


void Network::updateHiddentoOutputWeights() {}


char* Network::writeReport() {
    return std::string();
}


void Network::classify(float[InputNodes] inputs) {}

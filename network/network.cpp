/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 */

#include <random>

#include "network.hpp"

Network::Network(int InputNodes,
                 int HiddenNodes,
                 int OutputNodes,
                 float LearningRate,
                 float Momentum,
                 float InitialWeightMax,
                 std::mt19937 m_mt) {

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
}


void Network::initialiseHiddenWeights() {
    for (int i = 0; i < HiddenNodes; i++) {
        for (int j = 0; j <= InputNodes; j++) {
            ChangeHiddenWeights[j][i] = 0.0;
            Rando = dist(m_mt);
            HiddenWeights[j][i] = Rando * InitialWeightMax;
        }
    }
}


void Network::initialiseOutputWeights() {
    for(int i = 0 ; i < OutputNodes ; i ++ ) {
        for(int j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;
            Rando = dist(m_mt);
            OutputWeights[j][i] = Rando * InitialWeightMax ;
        }
    }
}


void Network::trainNetwork(float inputs[]) {

}


void Network::computeHiddenLayerActivations() {}


void Network::computeOutputLayerActivations() {}


void Network::backpropogateErrors() {}


void Network::updateInnerToHiddenWeights() {}


void Network::updateHiddentoOutputWeights() {}


char* Network::writeReport() {
    return 0;
}


void Network::classify(float inputs[]) {}

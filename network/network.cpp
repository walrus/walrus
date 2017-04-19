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


void Network::trainNetwork(float inputs[], float targets[]) {

}


void Network::computeHiddenLayerActivations(float inputs[]) {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        AccumulatedInput = HiddenWeights[InputNodes][i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) {
            AccumulatedInput += inputs[j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = float(1.0/(1.0 + exp(-AccumulatedInput))) ;
    }
}


void Network::computeOutputLayerActivations(float targets[]) {
    for(int i = 0 ; i < OutputNodes ; i++ ) {
        AccumulatedInput = OutputWeights[HiddenNodes][i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            AccumulatedInput += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = float(1.0/(1.0 + exp(-AccumulatedInput))) ;
        OutputDelta[i] = (targets[i] - Output[i]) * Output[i] * (1.0f - Output[i]) ;
        ErrorRate += 0.5 * (targets[i] - Output[i]) * (targets[i] - Output[i]) ;
    }
}


void Network::backpropogateErrors() {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        AccumulatedInput = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            AccumulatedInput += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = float(AccumulatedInput * Hidden[i] * (1.0 - Hidden[i])) ;
    }
}


void Network::updateInnerToHiddenWeights(float inputs[]) {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) {
            ChangeHiddenWeights[j][i] = LearningRate * inputs[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
            HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
    }
}


void Network::updateHiddentoOutputWeights() {
    for(int i = 0 ; i < OutputNodes ; i ++ ) {
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
            OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
    }
}


char* Network::writeReport() {
    return 0;
}


void Network::classify(float inputs[]) {}

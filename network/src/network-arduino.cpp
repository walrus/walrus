/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 *
 * Network_A is the lightweight version that will run on an Arduino.
 * It lacks functions for training the network and will rely on Flash memory rather than RAM.
 */

#include <random>

#include "network-arduino.hpp"

Network_A::Network_A(int numInputNodes,
                     int numHiddenNodes,
                     int numOutputNodes,
                     float learningRate,
                     float momentum,
                     float initialWeightMax):
        numInputNodes(numInputNodes),
        numHiddenNodes(numHiddenNodes),
        numOutputNodes(numOutputNodes),
        learningRate(learningRate),
        momentum(momentum),
        initialWeightMax(initialWeightMax) {

    accumulatedInput = 0.0f;

    hiddenNodes.resize(numHiddenNodes);
    outputNodes.resize(numOutputNodes);

    hiddenWeights.resize(numInputNodes+1, std::vector<float>(numHiddenNodes));
    outputWeights.resize(numHiddenNodes+1, std::vector<float>(numOutputNodes));
}

/*
 * Compute the activations of the hidden layer nodes from the given inputs
 */
void Network_A::computeHiddenLayerActivations(std::vector<float> inputs) {
    for(int i = 0 ; i < numHiddenNodes; i++ ) {
        accumulatedInput = hiddenWeights[numInputNodes][i] ;
        for(int j = 0 ; j < numInputNodes; j++ ) {
            accumulatedInput += inputs[j] * hiddenWeights[j][i] ;
        }
        hiddenNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


/*
 * Compute the activations of the hidden layer nodes from the current state of the hidden nodes.
 */
void Network_A::computeOutputLayerActivations() {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        accumulatedInput = outputWeights[numHiddenNodes][i] ;
        for(int j = 0 ; j < numHiddenNodes ; j++ ) {
            accumulatedInput += hiddenNodes[j] * outputWeights[j][i] ;
        }
        outputNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


/*
 * Using the current state of the network, attempt to classify the given input pattern,
 * and return a pointer to an array containing the predicted output.
 * The desired output for the function must be passed in.
 */
std::vector<float> Network_A::classify(std::vector<float> inputs) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations();
    std::vector<float> classification= outputNodes;
    return classification;
}


/*
 *  Set both sets of weights using pre calculated vectors.
 */
void Network_A::loadWeights(std::vector<std::vector<float>> hiddenWeights, std::vector<std::vector<float>> outputWeights) {
    setHiddenWeights(hiddenWeights);
    setOutputWeights(outputWeights);
}

int Network_A::getNumInputNodes() const {
    return numInputNodes;
}


int Network_A::getNumHiddenNodes() const {
    return numHiddenNodes;
}


int Network_A::getNumOutputNodes() const {
    return numOutputNodes;
}


float Network_A::getLearningRate() const {
    return learningRate;
}


float Network_A::getMomentum() const {
    return momentum;
}


float Network_A::getInitialWeightMax() const {
    return initialWeightMax;
}



float Network_A::getAccumulatedInput() const {
    return accumulatedInput;
}


const std::vector<float> Network_A::getHiddenNodes() const {
    return hiddenNodes;
}


const std::vector<float> Network_A::getOutputNodes() const {
    return outputNodes;
}


const std::vector<std::vector<float>> Network_A::getHiddenWeights() const {
    return hiddenWeights;
}


const std::vector<std::vector<float>> Network_A::getOutputWeights() const {
    return outputWeights;
}


void Network_A::setHiddenWeights(std::vector<std::vector<float>> hiddenWeights) {
    Network_A::hiddenWeights = hiddenWeights;
}


void Network_A::setOutputWeights(std::vector<std::vector<float>> outputWeights) {
    Network_A::outputWeights = outputWeights;
}

/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 */

#include <random>

#include "network.hpp"

Network::Network(std::mt19937 m_mt): m_mt(std::random_device()()) {

    dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    trainingCycle = 0;
    randomFloat = 0.0f;
    errorRate = 1.0f;
    accumulatedInput = 0.0f;

    initialiseHiddenWeights();
    initialiseOutputWeights();
}


/*
 * Initialise hiddenWeights to random values
 * Initialise hiddenWeightsChanges to zero
 * Use when setting up a new, untrained network
 */
void Network::initialiseHiddenWeights() {
    for (int i = 0; i < numHiddenNodes; i++) {
        for (int j = 0; j <= numInputNodes; j++) {
            hiddenWeightsChanges[j][i] = 0.0;
            randomFloat = dist(m_mt);
            hiddenWeights[j][i] = randomFloat * initialWeightMax;
        }
    }
}


/*
 * Initialise outputWeights to random values
 * Initialise outputWeightsChanges to zero
 * Use when setting up a new, untrained network
 */
void Network::initialiseOutputWeights() {
    for(int i = 0 ; i < numOutputNodes ; i ++ ) {
        for(int j = 0 ; j <= numHiddenNodes ; j++ ) {
            outputWeightsChanges[j][i] = 0.0 ;
            randomFloat = dist(m_mt);
            outputWeights[j][i] = randomFloat * initialWeightMax ;
        }
    }
}


/*
 * Train the network on a single pattern and return the error rate post training
 */
float Network::trainNetwork(float inputs[numInputNodes], float targets[numOutputNodes]) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations();

    computeErrors(targets);
    backpropagateErrors();

    updateHiddenWeights(inputs);
    updateOutputWeights();

    trainingCycle++;
    return errorRate;
}


/*
 * Compute the activations of the hidden layer nodes from the given inputs
 */
void Network::computeHiddenLayerActivations(float inputs[numInputNodes]) {
    for(int i = 0 ; i < numHiddenNodes ; i++ ) {
        accumulatedInput = hiddenWeights[numInputNodes][i] ;
        for(int j = 0 ; j < numInputNodes ; j++ ) {
            accumulatedInput += inputs[j] * hiddenWeights[j][i] ;
        }
        hiddenNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


/*
 * Compute the activations of the hidden layer nodes from the current state of the hidden nodes,
 * then compute the output errors and overall error rate
 */
void Network::computeOutputLayerActivations() {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        accumulatedInput = outputWeights[numHiddenNodes][i] ;
        for(int j = 0 ; j < numHiddenNodes ; j++ ) {
            accumulatedInput += hiddenNodes[j] * outputWeights[j][i] ;
        }
        outputNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


void Network::computeErrors(float targets[numOutputNodes]) {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        outputNodesDeltas[i] = (targets[i] - outputNodes[i]) * outputNodes[i] * (1.0f - outputNodes[i]);
        errorRate += 0.5 * (targets[i] - outputNodes[i]) * (targets[i] - outputNodes[i]);
    }
}


/*
 *  Backpropagate the output layer errors to the hidden layer
 */
void Network::backpropagateErrors() {
    for(int i = 0 ; i < numHiddenNodes ; i++ ) {
        accumulatedInput = 0.0 ;
        for(int j = 0 ; j < numOutputNodes ; j++ ) {
            accumulatedInput += outputWeights[i][j] * outputNodesDeltas[j] ;
        }
        hiddenNodesDeltas[i] = float(accumulatedInput * hiddenNodes[i] * (1.0 - hiddenNodes[i])) ;
    }
}


/*
 *  Using the backpropagated errors, update the weights of the hidden nodes
 */
void Network::updateHiddenWeights(float *inputs) {
    for(int i = 0 ; i < numHiddenNodes ; i++ ) {
        hiddenWeightsChanges[numInputNodes][i] = learningRate * hiddenNodesDeltas[i] + momentum * hiddenWeightsChanges[numInputNodes][i] ;
        hiddenWeights[numInputNodes][i] += hiddenWeightsChanges[numInputNodes][i] ;
        for(int j = 0 ; j < numInputNodes ; j++ ) {
            hiddenWeightsChanges[j][i] = learningRate * inputs[j] * hiddenNodesDeltas[i] + momentum * hiddenWeightsChanges[j][i];
            hiddenWeights[j][i] += hiddenWeightsChanges[j][i] ;
        }
    }
}


/*
 *  Using the backpropagated errors, update the weights of the hidden nodes
 */
void Network::updateOutputWeights() {
    for(int i = 0 ; i < numOutputNodes ; i ++ ) {
        outputWeightsChanges[numHiddenNodes][i] = learningRate * outputNodesDeltas[i] + momentum * outputWeightsChanges[numHiddenNodes][i] ;
        outputWeights[numHiddenNodes][i] += outputWeightsChanges[numHiddenNodes][i] ;
        for(int j = 0 ; j < numHiddenNodes ; j++ ) {
            outputWeightsChanges[j][i] = learningRate * hiddenNodes[j] * outputNodesDeltas[i] + momentum * outputWeightsChanges[j][i] ;
            outputWeights[j][i] += outputWeightsChanges[j][i] ;
        }
    }
}


/*
 * outputNodes the current training cycle and error rate as a string for display or logging
 */
std::string Network::writeReport() {
    return "Training cycle: " + std::to_string(trainingCycle) + ". Error rate: " + std::to_string(errorRate);
}


/*
 * Using the current state of the network, attempt to classify the given input pattern,
 * and return a pointer to an array containing the predicted output.
 * The desired output for the function must be passed in.
 */
float * Network::classify(float inputs[numInputNodes], float outputsDestination[numOutputNodes]) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations();

    std::copy(std::begin(outputNodes), std::end(outputNodes), outputsDestination);

    return outputsDestination;
}


const int Network::getNumInputNodes() {
    return numInputNodes;
}


const int Network::getNumHiddenNodes() {
    return numHiddenNodes;
}


const int Network::getNumOutputNodes() {
    return numOutputNodes;
}


float Network::getLearningRate() const {
    return learningRate;
}


float Network::getMomentum() const {
    return momentum;
}


float Network::getInitialWeightMax() const {
    return initialWeightMax;
}


long Network::getTrainingCycle() const {
    return trainingCycle;
}


float Network::getErrorRate() const {
    return errorRate;
}


const float *Network::getHiddenNodes() const {
    return hiddenNodes;
}


const float *Network::getOutputNodes() const {
    return outputNodes;
}


const float *Network::getHiddenNodesDeltas() const {
    return hiddenNodesDeltas;
}


const float *Network::getOutputNodesDeltas() const {
    return outputNodesDeltas;
}

void Network::setLearningRate(float learningRate) {
    Network::learningRate = learningRate;
}

void Network::setMomentum(float momentum) {
    Network::momentum = momentum;
}

void Network::setInitialWeightMax(float initialWeightMax) {
    Network::initialWeightMax = initialWeightMax;
}
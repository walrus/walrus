/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 *
 * Network_L is the full featured version that will run on a Linux machine.
 */

#include <random>

#include "network-linux.hpp"

Network_L::Network_L(int numInputNodes,
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
        initialWeightMax(initialWeightMax),
        m_mt(std::random_device()()) {

    dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    trainingCycle = 0;
    randomFloat = 0.0f;
    errorRate = 0.0f;
    accumulatedInput = 0.0f;

    hiddenNodes.resize(numHiddenNodes);
    outputNodes.resize(numOutputNodes);

    hiddenWeights.resize(numInputNodes+1, std::vector<float>(numHiddenNodes));
    outputWeights.resize(numHiddenNodes+1, std::vector<float>(numOutputNodes));

    hiddenNodesDeltas.resize(numHiddenNodes);
    outputNodesDeltas.resize(numOutputNodes);

    hiddenWeightsChanges.resize(numInputNodes+1, std::vector<float>(numHiddenNodes));
    outputWeightsChanges.resize(numHiddenNodes+1, std::vector<float>(numOutputNodes));

    initialiseHiddenWeights();
    initialiseOutputWeights();
}


/*
 * Initialise hiddenWeights to random values
 * Initialise hiddenWeightsChanges to zero
 * Use when setting up a new, untrained network
 */
void Network_L::initialiseHiddenWeights() {
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
void Network_L::initialiseOutputWeights() {
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
float Network_L::trainNetwork(std::vector<float> inputs, std::vector<float> targets) {
    errorRate = 0.0f;
    accumulatedInput = 0.0f;

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
void Network_L::computeHiddenLayerActivations(std::vector<float> inputs) {
    for(int i = 0 ; i < numHiddenNodes; i++ ) {
        accumulatedInput = hiddenWeights[numInputNodes][i] ;
        for(int j = 0 ; j < numInputNodes; j++ ) {
            accumulatedInput += inputs[j] * hiddenWeights[j][i] ;
        }
        hiddenNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


/*
 * Compute the activations of the hidden layer nodes from the current state of the hidden nodes,
 * then compute the output errors and overall error rate
 */
void Network_L::computeOutputLayerActivations() {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        accumulatedInput = outputWeights[numHiddenNodes][i] ;
        for(int j = 0 ; j < numHiddenNodes ; j++ ) {
            accumulatedInput += hiddenNodes[j] * outputWeights[j][i] ;
        }
        outputNodes[i] = float(1.0/(1.0 + exp(-accumulatedInput))) ;
    }
}


void Network_L::computeErrors(std::vector<float> targets) {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        outputNodesDeltas[i] = (targets[i] - outputNodes[i]) * outputNodes[i] * (1.0f - outputNodes[i]);
        errorRate += 0.5 * (targets[i] - outputNodes[i]) * (targets[i] - outputNodes[i]);
    }
}


/*
 *  Backpropagate the output layer errors to the hidden layer
 */
void Network_L::backpropagateErrors() {
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
void Network_L::updateHiddenWeights(std::vector<float> inputs) {
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
void Network_L::updateOutputWeights() {
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
 * outputs the current training cycle and error rate as a string for display or logging
 */
std::string Network_L::writeReport() {
    return "Training cycle: " + std::to_string(trainingCycle) + ". Error rate: " + std::to_string(errorRate);
}


/*
 * Using the current state of the network, attempt to classify the given input pattern,
 * and return a pointer to an array containing the predicted output.
 * The desired output for the function must be passed in.
 */
std::vector<float> Network_L::classify(std::vector<float> inputs) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations();
    std::vector<float> classification= outputNodes;
    return classification;
}


/*
 *  Set both sets of weights using pre calculated vectors.
 */
void Network_L::loadWeights(std::vector<std::vector<float>> hiddenWeights, std::vector<std::vector<float>> outputWeights) {
    setHiddenWeights(hiddenWeights);
    setOutputWeights(outputWeights);
}

int Network_L::getNumInputNodes() const {
    return numInputNodes;
}


int Network_L::getNumHiddenNodes() const {
    return numHiddenNodes;
}


int Network_L::getNumOutputNodes() const {
    return numOutputNodes;
}


float Network_L::getLearningRate() const {
    return learningRate;
}


float Network_L::getMomentum() const {
    return momentum;
}


float Network_L::getInitialWeightMax() const {
    return initialWeightMax;
}


long Network_L::getTrainingCycle() const {
    return trainingCycle;
}


float Network_L::getRandomFloat() const {
    return randomFloat;
}


float Network_L::getErrorRate() const {
    return errorRate;
}


float Network_L::getAccumulatedInput() const {
    return accumulatedInput;
}


const std::vector<float> Network_L::getHiddenNodes() const {
    return hiddenNodes;
}


const std::vector<float> Network_L::getOutputNodes() const {
    return outputNodes;
}


const std::vector<float> Network_L::getHiddenNodesDeltas() const {
    return hiddenNodesDeltas;
}


const std::vector<float> Network_L::getOutputNodesDeltas() const {
    return outputNodesDeltas;
}


const std::vector<std::vector<float>> Network_L::getHiddenWeights() const {
    return hiddenWeights;
}


const std::vector<std::vector<float>> Network_L::getOutputWeights() const {
    return outputWeights;
}


const std::vector<std::vector<float>> Network_L::getHiddenWeightsChanges() const {
    return hiddenWeightsChanges;
}


const std::vector<std::vector<float>> Network_L::getOutputWeightsChanges() const {
    return outputWeightsChanges;
}


void Network_L::setLearningRate(float learningRate) {
    Network_L::learningRate = learningRate;
}


void Network_L::setMomentum(float momentum) {
    Network_L::momentum = momentum;
}


void Network_L::setInitialWeightMax(float initialWeightMax) {
    Network_L::initialWeightMax = initialWeightMax;
}


void Network_L::setHiddenWeights(std::vector<std::vector<float>> hiddenWeights) {
    Network_L::hiddenWeights = hiddenWeights;
}


void Network_L::setOutputWeights(std::vector<std::vector<float>> outputWeights) {
    Network_L::outputWeights = outputWeights;
}

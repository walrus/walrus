/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 */

#include <random>

#include "network.hpp"

Network::Network(std::mt19937 m_mt): m_mt(std::random_device()()) {

    dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    TrainingCycle = 0;
    RandomFloat = 0.0f;
    ErrorRate = 1.0f;
    AccumulatedInput = 0.0f;
}


/*
 * Initialise HiddenWeights to random values
 * Initialise ChangeHiddenWeights to zero
 * Use when setting up a new, untrained network
 */
void Network::initialiseHiddenWeights() {
    for (int i = 0; i < HiddenNodes; i++) {
        for (int j = 0; j <= InputNodes; j++) {
            ChangeHiddenWeights[j][i] = 0.0;
            RandomFloat = dist(m_mt);
            HiddenWeights[j][i] = RandomFloat * InitialWeightMax;
        }
    }
}


/*
 * Initialise OutputWeights to random values
 * Initialise ChangeOutputWeights to zero
 * Use when setting up a new, untrained network
 */
void Network::initialiseOutputWeights() {
    for(int i = 0 ; i < OutputNodes ; i ++ ) {
        for(int j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;
            RandomFloat = dist(m_mt);
            OutputWeights[j][i] = RandomFloat * InitialWeightMax ;
        }
    }
}


/*
 * Train the network on a single pattern and return the error rate post training
 */
float Network::trainNetwork(float inputs[InputNodes], float targets[OutputNodes]) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations(targets);
    backpropagateErrors();
    updateHiddenWeights(inputs);
    updateOutputWeights();

    TrainingCycle++;
    return ErrorRate;
}

/*
 * Compute the activations of the hidden layer nodes from the given inputs
 */
void Network::computeHiddenLayerActivations(float inputs[InputNodes]) {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        AccumulatedInput = HiddenWeights[InputNodes][i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) {
            AccumulatedInput += inputs[j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = float(1.0/(1.0 + exp(-AccumulatedInput))) ;
    }
}

/*
 * Compute the activations of the hidden layer nodes from the current state of the hidden nodes,
 * then compute the output errors and overall error rate
 */
void Network::computeOutputLayerActivations(float targets[OutputNodes]) {
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


/*
 *  Backpropagate the output layer errors to the hidden layer
 */
void Network::backpropagateErrors() {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        AccumulatedInput = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            AccumulatedInput += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = float(AccumulatedInput * Hidden[i] * (1.0 - Hidden[i])) ;
    }
}


/*
 *  Using the backpropagated errors, update the weights of the hidden nodes
 */
void Network::updateHiddenWeights(float *inputs) {
    for(int i = 0 ; i < HiddenNodes ; i++ ) {
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for(int j = 0 ; j < InputNodes ; j++ ) {
            ChangeHiddenWeights[j][i] = LearningRate * inputs[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
            HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
    }
}


/*
 *  Using the backpropagated errors, update the weights of the hidden nodes
 */
void Network::updateOutputWeights() {
    for(int i = 0 ; i < OutputNodes ; i ++ ) {
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
            OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
    }
}

/*
 * Output the current training cycle and error rate as a string for display or logging
 */
std::string Network::writeReport() {
    return "Training cycle: " + std::to_string(TrainingCycle) + ". Error rate: " + std::to_string(ErrorRate);
}

/*
 * Using the current state of the network, attempt to classify the given input pattern,
 * and return a pointer to an array containing the predicted output
 */
float * Network::classify(float inputs[InputNodes]) {
    return new float[OutputNodes];
}

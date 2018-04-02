/*
 * A C++ library reworking of ArduinoANN - An artificial neural network for the Arduino
 * See robotics.hobbizine.com/arduinoann.html for more information about ArduinoANN.
 *
 * Network_L is the full featured version that will run on a Linux machine.
 */

#include <random>
#include <iostream>

#include "network-linux.hpp"

template class Network_L<float>;

template <class T>
Network_L<T>::Network_L(int numInputNodes,
                     int numHiddenNodes,
                     int numOutputNodes,
                     float learningRate,
                     float momentum,
                     float initialWeightMax,
                     long trainingCycle):
                     numInputNodes(numInputNodes),
                     numHiddenNodes(numHiddenNodes),
                     numOutputNodes(numOutputNodes),
                     learningRate(learningRate),
                     momentum(momentum),
                     initialWeightMax(initialWeightMax),
                     trainingCycle(trainingCycle),
                     m_mt(std::random_device()()) {

    dist = std::uniform_real_distribution<T>(-1.0f, 1.0f);

    randomFloat = 0.0f;
    errorRate = 0.0;
    accumulatedInput = 0.0f;

    hiddenActivationFunction = ActivationFunction::Sigmoid;
    outputActivationFunction = ActivationFunction::Sigmoid;
    errorFunction = ErrorFunction::SumSquared;

    hiddenNodes.resize(numHiddenNodes);
    outputNodes.resize(numOutputNodes);

    hiddenWeights.resize(numInputNodes+1, std::vector<T>(numHiddenNodes));
    outputWeights.resize(numHiddenNodes+1, std::vector<T>(numOutputNodes));

    hiddenNodesDeltas.resize(numHiddenNodes);
    outputNodesDeltas.resize(numOutputNodes);

    hiddenWeightsChanges.resize(numInputNodes+1, std::vector<T>(numHiddenNodes));
    outputWeightsChanges.resize(numHiddenNodes+1, std::vector<T>(numOutputNodes));

    initialiseHiddenWeights();
    initialiseOutputWeights();
}


/*
 * Initialise hiddenWeights to random values
 * Initialise hiddenWeightsChanges to zero
 * Use when setting up a new, untrained network
 */
template <class T>
void Network_L<T>::initialiseHiddenWeights() {
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
template <class T>
void Network_L<T>::initialiseOutputWeights() {
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
template <class T>
T Network_L<T>::trainNetwork(std::vector<T> inputs, std::vector<T> targets) {
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
 * Compute the activation for a single node using the selected activation function
 */
template <class T>
T Network_L<T>::computeActivation(T accumulatedInput, ActivationFunction af) {
    if (af == ActivationFunction::Sigmoid) {
        return float(1.0/(1.0 + exp(-accumulatedInput))) ;
    } else if (af == ActivationFunction::ReLu) {
        return std::max(0.0f, accumulatedInput);
    } else if (af == ActivationFunction::SoftMax) {
        return exp(accumulatedInput);
    } else {
        // Default to linear if for some reason activation is not specified
        return accumulatedInput;
    }
}

/*
 * Compute the activations of the hidden layer nodes from the given inputs
 */
template <class T>
void Network_L<T>::computeHiddenLayerActivations(std::vector<T> inputs) {
    float sumHidden = 0;
    for(int i = 0 ; i < numHiddenNodes; i++ ) {
        accumulatedInput = hiddenWeights[numInputNodes][i] ;
        for(int j = 0 ; j < numInputNodes; j++ ) {
            accumulatedInput += inputs[j] * hiddenWeights[j][i] ;
        }
        hiddenNodes[i] = computeActivation(accumulatedInput, hiddenActivationFunction);
        sumHidden += hiddenNodes[i];
    }
    // If we're using SoftMax then we need to divide each output node's output by their sum
    if (hiddenActivationFunction == ActivationFunction::SoftMax) {
        for (int i = 0; i < numHiddenNodes; i++) {
            hiddenNodes[i] = hiddenNodes[i] / sumHidden;
        }
    }
}


/*
 * Compute the activations of the output layer nodes from the current state of the hidden nodes,
 * then compute the output errors and overall error rate
 */
template <class T>
void Network_L<T>::computeOutputLayerActivations() {
    float sumOutputs = 0;
    for(int i = 0; i < numOutputNodes; i++ ) {
        accumulatedInput = outputWeights[numHiddenNodes][i] ;
        for(int j = 0; j < numHiddenNodes; j++ ) {
            accumulatedInput += hiddenNodes[j] * outputWeights[j][i] ;
        }
        outputNodes[i] = computeActivation(accumulatedInput, outputActivationFunction);
        sumOutputs += outputNodes[i];
    }
    // If using SoftMax then it is necessary to divide each output node's output by their sum
    if (outputActivationFunction == ActivationFunction::SoftMax) {
        for (int i = 0; i < numOutputNodes; i++) {
            outputNodes[i] = outputNodes[i] / sumOutputs;
        }
    }
}


/*
 *  Compute the delta for a single output node
 */
template <class T>
T Network_L<T>::computeDelta(T target, T output) {
    if (outputActivationFunction == ActivationFunction::Sigmoid
            && errorFunction == ErrorFunction::SumSquared) {
        return (target - output) * output * (1.0f - output);
    } else if (outputActivationFunction == ActivationFunction::ReLu
               || errorFunction == ErrorFunction::CrossEntropy) {
        return target - output;
    }
}


/*
 *  Compute the error rate using the selected error function
 */
template <class T>
T Network_L<T>::computeErrorRate(T target, T output) {
    if (errorFunction == ErrorFunction::SumSquared) {
        return 0.5 * (target - output) * (target - output);
    } else if (errorFunction == ErrorFunction::CrossEntropy) {
        return -1.0 * (target * log(output) + (1.0f - target) * log(1.0f - output));
    }
}


/*
 *  Compute the errors for the output layer
 */
template <class T>
void Network_L<T>::computeErrors(std::vector<T> targets) {
    for(int i = 0 ; i < numOutputNodes ; i++ ) {
        outputNodesDeltas[i] = computeDelta(targets[i], outputNodes[i]);
        errorRate += computeErrorRate(targets[i], outputNodes[i]);
    }
}


/*
 *  Backpropagate the output layer errors to the hidden layer
 */
template <class T>
void Network_L<T>::backpropagateErrors() {
    for(int i = 0 ; i < numHiddenNodes ; i++ ) {
        accumulatedInput = 0.0 ;
        for(int j = 0 ; j < numOutputNodes ; j++ ) {
            accumulatedInput += outputWeights[i][j] * outputNodesDeltas[j];
        }
        hiddenNodesDeltas[i] = float(accumulatedInput * hiddenNodes[i] * (1.0 - hiddenNodes[i])) ;
    }
}


/*
 *  Using the backpropagated errors, update the weights of the hidden nodes
 */
template <class T>
void Network_L<T>::updateHiddenWeights(std::vector<T> inputs) {
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
template <class T>
void Network_L<T>::updateOutputWeights() {
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
template <class T>
std::string Network_L<T>::writeReport() {
    return "Training cycle: " + std::to_string(trainingCycle) + ". Error rate: " + std::to_string(errorRate);
}


/*
 * Using the current state of the network, attempt to classify the given input pattern,
 * and return a pointer to an array containing the predicted output.
 * The desired output for the function must be passed in.
 */
template <class T>
std::vector<T> Network_L<T>::classify(std::vector<T> inputs) {
    computeHiddenLayerActivations(inputs);
    computeOutputLayerActivations();
    std::vector<T> classification= outputNodes;
    return classification;
}


/*
 *  Set both sets of weights using pre calculated vectors.
 */
template <class T>
void Network_L<T>::loadWeights(std::vector<std::vector<T>> hiddenWeights, std::vector<std::vector<T>> outputWeights) {
    setHiddenWeights(hiddenWeights);
    setOutputWeights(outputWeights);
}


template <class T>
int Network_L<T>::getNumInputNodes() const {
    return numInputNodes;
}


template <class T>
int Network_L<T>::getNumHiddenNodes() const {
    return numHiddenNodes;
}


template <class T>
int Network_L<T>::getNumOutputNodes() const {
    return numOutputNodes;
}


template <class T>
float Network_L<T>::getLearningRate() const {
    return learningRate;
}


template <class T>
float Network_L<T>::getMomentum() const {
    return momentum;
}


template <class T>
float Network_L<T>::getInitialWeightMax() const {
    return initialWeightMax;
}


template <class T>
long Network_L<T>::getTrainingCycle() const {
    return trainingCycle;
}


template <class T>
T Network_L<T>::getErrorRate() const {
    return errorRate;
}


template <class T>
T Network_L<T>::getAccumulatedInput() const {
    return accumulatedInput;
}


template <class T>
ActivationFunction Network_L<T>::getHiddenActivationFunction() const {
    return hiddenActivationFunction;
}


template <class T>
ActivationFunction Network_L<T>::getOutputActivationFunction() const {
    return outputActivationFunction;
}


template <class T>
ErrorFunction Network_L<T>::getErrorFunction() const {
    return errorFunction;
}


template <class T>
const std::vector<T> Network_L<T>::getHiddenNodes() const {
    return hiddenNodes;
}


template <class T>
const std::vector<T> Network_L<T>::getOutputNodes() const {
    return outputNodes;
}


template <class T>
const std::vector<T> Network_L<T>::getHiddenNodesDeltas() const {
    return hiddenNodesDeltas;
}


template <class T>
const std::vector<T> Network_L<T>::getOutputNodesDeltas() const {
    return outputNodesDeltas;
}


template <class T>
const std::vector<std::vector<T>> Network_L<T>::getHiddenWeights() const {
    return hiddenWeights;
}


template <class T>
const std::vector<std::vector<T>> Network_L<T>::getOutputWeights() const {
    return outputWeights;
}


template <class T>
const std::vector<std::vector<T>> Network_L<T>::getHiddenWeightsChanges() const {
    return hiddenWeightsChanges;
}


template <class T>
const std::vector<std::vector<T>> Network_L<T>::getOutputWeightsChanges() const {
    return outputWeightsChanges;
}


template <class T>
void Network_L<T>::setLearningRate(float learningRate) {
    Network_L<T>::learningRate = learningRate;
}


template <class T>
void Network_L<T>::setMomentum(float momentum) {
    Network_L<T>::momentum = momentum;
}


template <class T>
void Network_L<T>::setHiddenActivationFunction(ActivationFunction activationFunction) {
    Network_L<T>::hiddenActivationFunction = activationFunction;
}


template <class T>
void Network_L<T>::setOutputActivationFunction(ActivationFunction activationFunction) {
    Network_L<T>::outputActivationFunction = activationFunction;
}


template <class T>
void Network_L<T>::setErrorFunction(ErrorFunction errorFunction) {
    Network_L<T>::errorFunction = errorFunction;
}


template <class T>
void Network_L<T>::setHiddenWeights(std::vector<std::vector<T>> hiddenWeights) {
    Network_L<T>::hiddenWeights = hiddenWeights;
}


template <class T>
void Network_L<T>::setOutputWeights(std::vector<std::vector<T>> outputWeights) {
    Network_L<T>::outputWeights = outputWeights;
}


/*
 * Utility function to get an Activation Function from a string
 */
ActivationFunction stringToAF(std::string name) {
    if (name == "Sigmoid") {
        return ActivationFunction::Sigmoid;
    } else if (name == "ReLu") {
        return ActivationFunction::ReLu;
    } else if (name == "SoftMax") {
        return ActivationFunction::SoftMax;
    } else {
        std::cout << "Activation function not recognised: " << name << "\n";
        return ActivationFunction::Sigmoid; // Default to Sigmoid
    }
}


/*
 * Utility function to get the string representation of an Activation Function
 */
std::string aFToString(ActivationFunction af) {
    if (af == ActivationFunction::Sigmoid) {
        return "Sigmoid";
    } else if (af == ActivationFunction::ReLu) {
        return "ReLu";
    } else if (af == ActivationFunction::SoftMax) {
        return "SoftMax";
    }
}


/*
 * Utility function to get an Error Function from a string
 */
ErrorFunction stringToEF(std::string name) {
    if (name == "SumSquared") {
        return ErrorFunction::SumSquared;
    } else if (name == "CrossEntropy") {
        return ErrorFunction::CrossEntropy;
    } else {
        std::cout << "Error function not recognised: " << name << "\n";
        return ErrorFunction::SumSquared;
    }
}


/*
 * Utility function to get the string representation of an Error Function
 */
std::string eFToString(ErrorFunction ef) {
    if (ef == ErrorFunction::SumSquared) {
        return "SumSquared";
    } else if (ef == ErrorFunction::CrossEntropy) {
        return "CrossEntropy";
    }
}

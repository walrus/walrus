#ifndef NETWORK_L_H
#define NETWORK_L_H

#include <vector>
#include <random>

enum class ActivationFunction {Sigmoid, ReLu, SoftMax};

ActivationFunction stringToAF(std::string name);
std::string aFToString(ActivationFunction af);

enum class ErrorFunction {SumSquared, CrossEntropy};

ErrorFunction stringToEF(std::string name);
std::string eFToString(ErrorFunction ef);

template <class T> class Network_L {
T i; // Type template
private:
    const int numInputNodes;                                // AKA 'InputNodes' in the original code
    const int numHiddenNodes;                               // AKA 'HiddenNodes' in the original code
    const int numOutputNodes;                               // AKA 'OutputNodes' in the original code
    float learningRate;                                     // AKA 'LearningRate' in the original code
    float momentum;                                         // AKA 'Momentum' in the original code
    float initialWeightMax;                                 // AKA 'InitialWeightMax' in the original code

    long trainingCycle;                                     // AKA 'TrainingCycle' in the original code
    float randomFloat;                                      // AKA 'Rando' in the original code
    double errorRate;                                       // AKA 'Error' in the original code
    float accumulatedInput;                                 // AKA 'Accum' in the original code

    ActivationFunction hiddenActivationFunction;            // Activation function. Original code used Sigmoid
    ActivationFunction outputActivationFunction;            // Activation function. Original code used Sigmoid
    ErrorFunction  errorFunction;                           // Error function. Original code used SumSquared

    std::vector<T> hiddenNodes;                         // AKA 'Hidden' in the original code
    std::vector<T> outputNodes;                         // AKA 'Output' in the original code
    std::vector<std::vector<T>> hiddenWeights;          // AKA 'HiddenWeights' in the original code
    std::vector<std::vector<T>> outputWeights;          // AKA 'OutputWeights' in the original code
    std::vector<T> hiddenNodesDeltas;                   // AKA 'HiddenDelta' in the original code
    std::vector<T> outputNodesDeltas;                   // AKA 'OutputDelta' in the original code
    std::vector<std::vector<T>> hiddenWeightsChanges;   // AKA 'ChangeHiddenWeights' in the original code
    std::vector<std::vector<T>> outputWeightsChanges;   // AKA 'ChangeOutputWeights' in the original code

    std::mt19937 m_mt;                                      // Mersenne twister for random number generation
    std::uniform_real_distribution<T> dist;             // Distribution for random number generation

    void initialiseHiddenWeights();
    void initialiseOutputWeights();
    T computeActivation(T accumulatedInput, ActivationFunction af);
    void computeHiddenLayerActivations(std::vector<T> inputs);
    void computeOutputLayerActivations();
    T computeDelta(T target, T output);
    T computeErrorRate(T target, T output);
    void computeErrors(std::vector<T> targets);
    void backpropagateErrors();
    void updateHiddenWeights(std::vector<T> inputs);
    void updateOutputWeights();
    void setHiddenWeights(std::vector<std::vector<T>> hiddenWeights);
    void setOutputWeights(std::vector<std::vector<T>> outputWeights);

public:
    Network_L<T>(int numInputNodes,
              int numHiddenNodes,
              int numOutputNodes,
              float learningRate,
              float momentum,
              float initialWeightMax,
              long trainingCycle);
    T trainNetwork(std::vector<T> inputs,
                       std::vector<T> targets);
    std::string writeReport();
    std::vector<T> classify(std::vector<T> inputs);
    void loadWeights(std::vector<std::vector<T>> hiddenWeights,
                     std::vector<std::vector<T>> outputWeights);

    int getNumInputNodes() const;
    int getNumHiddenNodes() const;
    int getNumOutputNodes() const;
    float getLearningRate() const;
    float getMomentum() const;
    float getInitialWeightMax() const;
    long getTrainingCycle() const;
    T getErrorRate() const;
    T getAccumulatedInput() const;
    ActivationFunction getHiddenActivationFunction() const;
    ActivationFunction getOutputActivationFunction() const;
    ErrorFunction getErrorFunction() const;
    const std::vector<T> getHiddenNodes() const;
    const std::vector<T> getOutputNodes() const;
    const std::vector<T> getHiddenNodesDeltas() const;
    const std::vector<T> getOutputNodesDeltas() const;
    const std::vector<std::vector<T>> getHiddenWeights() const;
    const std::vector<std::vector<T>> getOutputWeights() const;
    const std::vector<std::vector<T>> getHiddenWeightsChanges() const;
    const std::vector<std::vector<T>> getOutputWeightsChanges() const;
    void setLearningRate(float learningRate);
    void setMomentum(float momentum);
    void setHiddenActivationFunction(ActivationFunction activationFunction);
    void setOutputActivationFunction(ActivationFunction activationFunction);
    void setErrorFunction(ErrorFunction errorFunction);
};

#endif // NETWORK_L_H
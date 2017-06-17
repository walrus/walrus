#ifndef NETWORK_L_H
#define NETWORK_L_H

#include <vector>
#include <random>

enum class ActivationFunction {Sigmoid, ReLu};

ActivationFunction stringToAF(std::string name);
std::string aFToString(ActivationFunction af);

enum class ErrorFunction {SumSquared, CrossEntropy};

ErrorFunction stringToEF(std::string name);
std::string eFToString(ErrorFunction ef);

class Network_L {
private:
    const int numInputNodes;                                // AKA 'InputNodes' in the original code
    const int numHiddenNodes;                               // AKA 'HiddenNodes' in the original code
    const int numOutputNodes;                               // AKA 'OutputNodes' in the original code
    float learningRate;                                     // AKA 'LearningRate' in the original code
    float momentum;                                         // AKA 'Momentum' in the original code
    float initialWeightMax;                                 // AKA 'InitialWeightMax' in the original code

    long trainingCycle;                                     // AKA 'TrainingCycle' in the original code
    float randomFloat;                                      // AKA 'Rando' in the original code
    float errorRate;                                        // AKA 'Error' in the original code
    float accumulatedInput;                                 // AKA 'Accum' in the original code

    ActivationFunction activationFunction;                  // Activation function. Original code used Sigmoid
    ErrorFunction  errorFunction;                           // Error function. Original code used SumSquared

    std::vector<float> hiddenNodes;                         // AKA 'Hidden' in the original code
    std::vector<float> outputNodes;                         // AKA 'Output' in the original code
    std::vector<std::vector<float>> hiddenWeights;          // AKA 'HiddenWeights' in the original code
    std::vector<std::vector<float>> outputWeights;          // AKA 'OutputWeights' in the original code
    std::vector<float> hiddenNodesDeltas;                   // AKA 'HiddenDelta' in the original code
    std::vector<float> outputNodesDeltas;                   // AKA 'OutputDelta' in the original code
    std::vector<std::vector<float>> hiddenWeightsChanges;   // AKA 'ChangeHiddenWeights' in the original code
    std::vector<std::vector<float>> outputWeightsChanges;   // AKA 'ChangeOutputWeights' in the original code

    std::mt19937 m_mt;                                      // Mersenne twister for random number generation
    std::uniform_real_distribution<float> dist;             // Distribution for random number generation

    void initialiseHiddenWeights();
    void initialiseOutputWeights();
    float computeActivation(float accumulatedInput);
    void computeHiddenLayerActivations(std::vector<float> inputs);
    void computeOutputLayerActivations();
    float computeDelta(float target, float output);
    float computeErrorRate(float target, float output);
    void computeErrors(std::vector<float> targets);
    void backpropagateErrors();
    void updateHiddenWeights(std::vector<float> inputs);
    void updateOutputWeights();
    void setHiddenWeights(std::vector<std::vector<float>> hiddenWeights);
    void setOutputWeights(std::vector<std::vector<float>> outputWeights);

public:
    Network_L(int numInputNodes,
              int numHiddenNodes,
              int numOutputNodes,
              float learningRate,
              float momentum,
              float initialWeightMax,
              long trainingCycle);
    float trainNetwork(std::vector<float> inputs,
                       std::vector<float> targets);
    std::string writeReport();
    std::vector<float> classify(std::vector<float> inputs);
    void loadWeights(std::vector<std::vector<float>> hiddenWeights,
                     std::vector<std::vector<float>> outputWeights);

    int getNumInputNodes() const;
    int getNumHiddenNodes() const;
    int getNumOutputNodes() const;
    float getLearningRate() const;
    float getMomentum() const;
    float getInitialWeightMax() const;
    long getTrainingCycle() const;
    float getRandomFloat() const;
    float getErrorRate() const;
    float getAccumulatedInput() const;
    ActivationFunction getActivationFunction() const;
    ErrorFunction getErrorFunction() const;
    const std::vector<float> getHiddenNodes() const;
    const std::vector<float> getOutputNodes() const;
    const std::vector<float> getHiddenNodesDeltas() const;
    const std::vector<float> getOutputNodesDeltas() const;
    const std::vector<std::vector<float>> getHiddenWeights() const;
    const std::vector<std::vector<float>> getOutputWeights() const;
    const std::vector<std::vector<float>> getHiddenWeightsChanges() const;
    const std::vector<std::vector<float>> getOutputWeightsChanges() const;
    void setLearningRate(float learningRate);
    void setMomentum(float momentum);
    void setInitialWeightMax(float initialWeightMax);
    void setActivationFunction(ActivationFunction activationFunction);
    void setErrorFunction(ErrorFunction errorFunction);
};

#endif // NETWORK_L_H
#ifndef NETWORK_A_H
#define NETWORK_A_H

#include <vector>
#include <random>

class Network_A {
private:
    const int numInputNodes;                    // AKA 'InputNodes' in the original code
    const int numHiddenNodes;                   // AKA 'HiddenNodes' in the original code
    const int numOutputNodes;                   // AKA 'OutputNodes' in the original code
    float learningRate;                         // AKA 'LearningRate' in the original code
    float momentum;                             // AKA 'Momentum' in the original code
    float initialWeightMax;                     // AKA 'InitialWeightMax' in the original code

    float accumulatedInput;                     // AKA 'Accum' in the original code

    std::vector<float> hiddenNodes;                  // AKA 'Hidden' in the original code
    std::vector<float> outputNodes;                  // AKA 'Output' in the original code
    std::vector<std::vector<float>> hiddenWeights;        // AKA 'HiddenWeights' in the original code
    std::vector<std::vector<float>> outputWeights;        // AKA 'OutputWeights' in the original code

    void computeHiddenLayerActivations(std::vector<float> inputs);
    void computeOutputLayerActivations();

    void setHiddenWeights(std::vector<std::vector<float>> hiddenWeights);
    void setOutputWeights(std::vector<std::vector<float>> outputWeights);


public:
    Network_A(int numInputNodes,
              int numHiddenNodes,
              int numOutputNodes,
              float learningRate,
              float momentum,
              float initialWeightMax);
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
    float getAccumulatedInput() const;
    const std::vector<float> getHiddenNodes() const;
    const std::vector<float> getOutputNodes() const;
    const std::vector<std::vector<float>> getHiddenWeights() const;
    const std::vector<std::vector<float>> getOutputWeights() const;
};

#endif // NETWORK_A_H
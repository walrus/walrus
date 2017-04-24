#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>

using std::vector;

#define NUM_INPUT_NODES 20;
#define NUM_HIDDEN_NODES 20;
#define NUM_OUTPUT_NODES 20;

#define DEFAULT_LEARNING_RATE 0.3;
#define DEFAULT_MOMENTUM 0.9;
#define DEFAULT_INITIAL_WEIGHT_MAX 0.5;

class Network {
    private:
        constexpr static int numInputNodes = NUM_INPUT_NODES;           // AKA 'InputNodes' in the original code
        constexpr static int numHiddenNodes = NUM_HIDDEN_NODES;         // AKA 'HiddenNodes' in the original code
        constexpr static int numOutputNodes = NUM_OUTPUT_NODES;         // AKA 'OutputNodes' in the original code
        float learningRate = DEFAULT_LEARNING_RATE;                     // AKA 'LearningRate' in the original code
        float momentum = DEFAULT_MOMENTUM;                              // AKA 'Momentum' in the original code
        float initialWeightMax = DEFAULT_INITIAL_WEIGHT_MAX;            // AKA 'InitialWeightMax' in the original code

        long trainingCycle;                                             // AKA 'TrainingCycle' in the original code
        float randomFloat;                                              // AKA 'Rando' in the original code
        float errorRate;                                                // AKA 'Error' in the original code
        float accumulatedInput;                                         // AKA 'Accum' in the original code

        vector<float> hiddenNodes;                                      // AKA 'Hidden' in the original code
        vector<float> outputNodes;                                      // AKA 'Output' in the original code
        vector<vector<float>> hiddenWeights;                            // AKA 'HiddenWeights' in the original code
        vector<vector<float>> outputWeights;                            // AKA 'OutputWeights' in the original code
        vector<float> hiddenNodesDeltas;                                // AKA 'HiddenDelta' in the original code
        vector<float> outputNodesDeltas;                                // AKA 'OutputDelta' in the original code
        vector<vector<float>> hiddenWeightsChanges;                     // AKA 'ChangeHiddenWeights' in the original code
        vector<vector<float>> outputWeightsChanges;                     // AKA 'ChangeOutputWeights' in the original code

        std::mt19937 m_mt;                                              // Mersenne twister for random number generation
        std::uniform_real_distribution<float> dist;                     // Distribution for random number generation

        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        void computeHiddenLayerActivations(vector<float> inputs);
        void computeOutputLayerActivations();
        void computeErrors(vector<float> targets);
        void backpropagateErrors();
        void updateHiddenWeights(vector<float> inputs);
        void updateOutputWeights();

    public:
        Network(std::mt19937 m_mt);
        float trainNetwork(vector<float> inputs, vector<float> targets);
        std::string writeReport();
        vector<float> classify(vector<float> inputs);

        static const int getNumInputNodes();
        static const int getNumHiddenNodes();
        static const int getNumOutputNodes();
        float getLearningRate() const;
        float getMomentum() const;
        float getInitialWeightMax() const;
        long getTrainingCycle() const;
        float getRandomFloat() const;
        float getErrorRate() const;
        float getAccumulatedInput() const;
        const vector<float> getHiddenNodes() const;
        const vector<float> getOutputNodes() const;
        const vector<float> getHiddenNodesDeltas() const;
        const vector<float> getOutputNodesDeltas() const;
        void setLearningRate(float learningRate);
        void setMomentum(float momentum);
        void setInitialWeightMax(float initialWeightMax);
};

#endif
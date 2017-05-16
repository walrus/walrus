#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <random>

using std::vector;

class Network {
    private:
        const int numInputNodes;                    // AKA 'InputNodes' in the original code
        const int numHiddenNodes;                   // AKA 'HiddenNodes' in the original code
        const int numOutputNodes;                   // AKA 'OutputNodes' in the original code
        float learningRate;                         // AKA 'LearningRate' in the original code
        float momentum;                             // AKA 'Momentum' in the original code
        float initialWeightMax;                     // AKA 'InitialWeightMax' in the original code

        long trainingCycle;                         // AKA 'TrainingCycle' in the original code
        float randomFloat;                          // AKA 'Rando' in the original code
        float errorRate;                            // AKA 'Error' in the original code
        float accumulatedInput;                     // AKA 'Accum' in the original code

        vector<float> hiddenNodes;                  // AKA 'Hidden' in the original code
        vector<float> outputNodes;                  // AKA 'Output' in the original code
        vector<vector<float>> hiddenWeights;        // AKA 'HiddenWeights' in the original code
        vector<vector<float>> outputWeights;        // AKA 'OutputWeights' in the original code
        vector<float> hiddenNodesDeltas;            // AKA 'HiddenDelta' in the original code
        vector<float> outputNodesDeltas;            // AKA 'OutputDelta' in the original code
        vector<vector<float>> hiddenWeightsChanges; // AKA 'ChangeHiddenWeights' in the original code
        vector<vector<float>> outputWeightsChanges; // AKA 'ChangeOutputWeights' in the original code

        std::mt19937 m_mt;                          // Mersenne twister for random number generation
        std::uniform_real_distribution<float> dist; // Distribution for random number generation

        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        void computeHiddenLayerActivations(vector<float> inputs);
        void computeOutputLayerActivations();
        void computeErrors(vector<float> targets);
        void backpropagateErrors();
        void updateHiddenWeights(vector<float> inputs);
        void updateOutputWeights();
        void setHiddenWeights(vector<vector<float>> hiddenWeights);
        void setOutputWeights(vector<vector<float>> outputWeights);

    public:
        Network(int numInputNodes,
                int numHiddenNodes,
                int numOutputNodes,
                float learningRate,
                float momentum,
                float initialWeightMax);
        float trainNetwork(vector<float> inputs,
                           vector<float> targets);
        std::string writeReport();
        vector<float> classify(vector<float> inputs);
        void loadWeights(vector<vector<float>> hiddenWeights,
                         vector<vector<float>> outputWeights);

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
        const vector<float> getHiddenNodes() const;
        const vector<float> getOutputNodes() const;
        const vector<float> getHiddenNodesDeltas() const;
        const vector<float> getOutputNodesDeltas() const;
        const vector<vector<float>> getHiddenWeights() const;
        const vector<vector<float>> getOutputWeights() const;
        const vector<vector<float>> getHiddenWeightsChanges() const;
        const vector<vector<float>> getOutputWeightsChanges() const;
        void setLearningRate(float learningRate);
        void setMomentum(float momentum);
        void setInitialWeightMax(float initialWeightMax);
};

#endif
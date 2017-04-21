#ifndef NETWORK_H
#define NETWORK_H

class Network {
    private:
        constexpr static int numInputNodes = 20;                        // AKA 'InputNodes' in the original code
        constexpr static int numHiddenNodes = 20;                       // AKA 'HiddenNodes' in the original code
        constexpr static int numOutputNodes = 1;                        // AKA 'OutputNodes' in the original code
        float learningRate = 0.3;                                       // AKA 'LearningRate' in the original code
        float momentum = 0.9;                                           // AKA 'Momentum' in the original code
        float initialWeightMax = 0.5;                                   // AKA 'InitialWeightMax' in the original code

        long trainingCycle;                                             // AKA 'TrainingCycle' in the original code
        float randomFloat;                                              // AKA 'Rando' in the original code
        float errorRate;                                                // AKA 'Error' in the original code
        float accumulatedInput;                                         // AKA 'Accum' in the original code

        float hiddenNodes[numHiddenNodes];                              // AKA 'Hidden' in the original code
        float outputNodes[numOutputNodes];                              // AKA 'Output' in the original code
        float hiddenWeights[numInputNodes+1][numHiddenNodes];           // AKA 'HiddenWeights' in the original code
        float outputWeights[numHiddenNodes+1][numOutputNodes];          // AKA 'OutputWeights' in the original code
        float hiddenNodesDeltas[numHiddenNodes];                        // AKA 'HiddenDelta' in the original code
        float outputNodesDeltas[numOutputNodes];                        // AKA 'OutputDelta' in the original code
        float hiddenWeightsChanges[numInputNodes+1][numHiddenNodes];    // AKA 'ChangeHiddenWeights' in the original code
        float outputWeightsChanges[numHiddenNodes+1][numOutputNodes];   // AKA 'ChangeOutputWeights' in the original code

        std::mt19937 m_mt;                                              // Mersenne twister for random number generation
        std::uniform_real_distribution<float> dist;                     // Distribution for random number generation

        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        void computeHiddenLayerActivations(float inputs[numInputNodes]);
        void computeOutputLayerActivations();
        void computeErrors(float targets[numOutputNodes]);
        void backpropagateErrors();
        void updateHiddenWeights(float inputs[numInputNodes]);
        void updateOutputWeights();

    public:
        Network(std::mt19937 m_mt);
        float trainNetwork(float inputs[numInputNodes], float targets[numOutputNodes]);
        std::string writeReport();
        float * classify(float inputs[numInputNodes], float outputsDestination[numOutputNodes]);

        static const int getNumInputNodes();
        static const int getNumHiddenNodes();
        static const int getNumOutputNodes();
        float getLearningRate() const;
        float getMomentum() const;
        float getInitialWeightMax() const;
        long getTrainingCycle() const;
        float getErrorRate() const;
        const float *getHiddenNodes() const;
        const float *getOutputNodes() const;
        const float *getHiddenNodesDeltas() const;
        const float *getOutputNodesDeltas() const;

        void setLearningRate(float learningRate);
        void setMomentum(float momentum);
        void setInitialWeightMax(float initialWeightMax);
};

#endif
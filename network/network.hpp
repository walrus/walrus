#ifndef NETWORK_H
#define NETWORK_H

class Network {
    private:
        constexpr static int InputNodes = 20;
        constexpr static int HiddenNodes = 20;
        constexpr static int OutputNodes = 1;
        constexpr static float LearningRate = 0.3;
        constexpr static float Momentum = 0.9;
        constexpr static float InitialWeightMax = 0.5;

        long TrainingCycle;
        float RandomFloat;      // AKA 'Rando' in the original code
        float ErrorRate;        // AKA 'Error' in the original code
        float AccumulatedInput; // AKA 'Accum' in the original code

        float Hidden[HiddenNodes];
        float Output[OutputNodes];
        float HiddenWeights[InputNodes+1][HiddenNodes];
        float OutputWeights[HiddenNodes+1][OutputNodes];
        float HiddenDelta[HiddenNodes];
        float OutputDelta[OutputNodes];
        float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
        float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

        std::mt19937 m_mt;
        std::uniform_real_distribution<float> dist;

    public:
        Network(std::mt19937 m_mt);
        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        float trainNetwork(float inputs[InputNodes], float targets[OutputNodes]);
        void computeHiddenLayerActivations(float inputs[InputNodes]);
        void computeOutputLayerActivations(float targets[OutputNodes]);
        void backpropagateErrors();
        void updateHiddenWeights(float *inputs);
        void updateOutputWeights();
        std::string writeReport();
        float * classify(float *inputs);
};

#endif
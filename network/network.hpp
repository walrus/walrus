#ifndef NETWORK_H
#define NETWORK_H

class Network {
    private:
        const static int InputNodes = 20;
        const static int HiddenNodes = 20;
        const static int OutputNodes = 1;
        const static float LearningRate = 0.3;
        const static float Momentum = 0.9;
        const static float InitialWeightMax = 0.5;

        int ReportEvery1000;
        long TrainingCycle;
        float Rando;
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
        Network(int InputNodes,
                int HiddenNodes,
                int OutputNodes,
                float LearningRate,
                float Momentum,
                float InitialWeightMax,
                std::mt19937 m_mt);
        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        void trainNetwork(float inputs[], float targets[]);
        void computeHiddenLayerActivations(float inputs[]);
        void computeOutputLayerActivations(float targets[]);
        void backpropogateErrors();
        void updateInnerToHiddenWeights(float inputs[]);
        void updateHiddentoOutputWeights();
        char* writeReport();
        void classify(float inputs[]);
};

#endif
#ifndef NETWORK_H
#define NETWORK_H

class Network {
    private:
        const int InputNodes = 20;
        const int HiddenNodes = 20;
        const int OutputNodes = 1;
        const float LearningRate = 0.3;
        const float Momentum = 0.9;
        const float InitialWeightMax 0.5;

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

    public:
        Network(int InputNodes,
                int HiddenNodes,
                int OutputNodes,
                float LearningRate,
                float Momentum,
                float InitialWeightMax,
                float Success);
        void initialiseHiddenWeights();
        void initialiseOutputWeights();
        void trainNetwork(float[InputNodes] inputs);
        void computeHiddenLayerActivations();
        void computeOutputLayerActivations();
        void backpropogateErrors();
        void updateInnerToHiddenWeights();
        void updateHiddentoOutputWeights();
        char* writeReport();
        void classify(float[InputNodes] inputs);
}

#endif
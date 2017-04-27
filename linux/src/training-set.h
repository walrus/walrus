#ifndef TRAINING_SET_H
#define TRAINING_SET_H

/* Training set data type for exercise classification */

#include <vector>
using std::vector;

class TrainingSet {
    public:
        vector<vector<float>> inputs;
        vector<vector<float>> targets;
        TrainingSet();
};

TrainingSet *loadTrainingSet(std::string filename);

#endif // TRAINING_SET_H

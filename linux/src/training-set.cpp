#include "training-set.hpp"

TrainingSet::TrainingSet() {}

TrainingSet *loadTrainingSet(std::string filename) {
    return new TrainingSet();
}

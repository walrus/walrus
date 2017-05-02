#include "training-set.hpp"

TrainingSet::TrainingSet() {}

TrainingSet *loadTrainingSet(std::string filename) {
    TrainingSet* set = new TrainingSet();

    // Load and process file
    std::ifstream log_file (filename);
    std::string line;

    vector<float> currentInput;
    vector<float> currentTarget;
    // True if 'repetition end' has been seen, IE the numbers being seen are targets not inputs
    bool readingTargets = false;

    while (std::getline(log_file, line)) {
        if (line == "Repetition start") {
            if (currentTarget.size() > 0) {
                set->inputs.push_back(currentInput);
                set->targets.push_back(currentTarget);
            }
            currentInput.clear();
            currentTarget.clear();
            readingTargets = false;
        }else if (line == "Repetition end") {
            readingTargets = true;
        }else {
            if (readingTargets) {
                currentTarget.push_back(std::stof(line));
            }
            else {
                currentInput.push_back(std::stof(line));
            }
        }
    }

    // Load the final values of current input/target too
    if (readingTargets) {
        set->inputs.push_back(currentInput);
        set->targets.push_back(currentTarget);
    }

    return set;
}

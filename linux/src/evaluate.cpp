/*
 * Main evaluation program for the network.
 *
 * Run from command line as follows:
 *
 * train config_filename validationdir threshold
 *
 * Will validate the network on the contents of validationdir
 *
 * Threshold is the number above which a target is counted
 *
 * For example, with a threshold of 0.5,  [0.6, 0.3, 0.1] would return 0, while [0.4, 0.1, 0.1] would return 3
 *
 * Must be run from the linux/ directory
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <random>
#include <algorithm>

#include "../../network/src/network-linux.hpp"
#include "../../network/src/network-saveload-linux.hpp"
#include "training-set.hpp"

// Arguments
std::string config_file_location;
std::string validationdir;
float classificationThreshold = 0.5; // Default value

// Validation data
std::vector<std::vector<float>> validationTargets;
std::vector<std::vector<float>> validationOutputs;
std::vector<int> targetClassifications;
std::vector<int> outputClassifications;

// Confusion matrix & other metrics
std::vector<std::vector<int>> confusion;
std::vector<std::string> caLabels = {" Press up | ", "   Sit up | ", "    Lunge | ", "     None | "};

std::vector<int> predictions;  // Number of predicted classifications for each class
std::vector<int> actuals;      // Number of actual classifications for each class

std::vector<float> recalls;    // Recall of each class (correct predictions over actuals)
std::vector<float> precisions; // Precision of each class (correct predictions over predictions)
std::vector<float> fones;      // F1 measure of each class
float uar = 0;                 // Unweighted average recall

void validateSet(std::string filename, Network_L *network) {
    // Check the training file exists, and if it doesn't, exit
    std::ifstream check_logfile(filename);
    if (!check_logfile.good() || filename.find("_normalised") == std::string::npos) {
        check_logfile.close();
        std::cout << filename << " is an invalid log file, skipping.\n";
    } else {

        TrainingSet *set = loadTrainingSet(filename);

        for (int i = 0; i < set->inputs.size(); i++) {
            validationOutputs.push_back(network->classify(set->inputs[i]));
            validationTargets.push_back(set->targets[i]);

        }
    }
}

void validateDir(std::string dirname, Network_L *network) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (dirname.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == DT_REG &&
                std::string(ent->d_name).find("_normalised") != std::string::npos) {
                validateSet(dirname + std::string(ent->d_name), network);
            }else if (ent->d_type == DT_DIR &&
                      std::string(ent->d_name).find(".") == std::string::npos) {
                validateDir(dirname + std::string(ent->d_name) + "/", network);
            }
        }
        closedir (dir);
    } else {
        std::cout << "Could not open directory " << dirname << "\n";
    }
}

int main(int argc, char * argv[]) {
    // Parse arguments
    if (argc < 4) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 4) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    // Only show 2dp on standard output
    std::cout.precision(2);

    config_file_location = argv[1];
    validationdir = argv[2];
    classificationThreshold = std::stof(argv[3]);

    Network_L *network;

    // Check the network config file exists, and if it doesn't, create it.
    std::cout << "Checking network config file...";
    std::ifstream check_config(config_file_location);
    if (!check_config.is_open()) {
        std::cout << "not found, exiting\n";
        return 1;
    } else {
        std::cout << "found, loading network\n";
        // Load the network
        network = loadNetwork(config_file_location);
    }

    std::cout << "Validating...\n";
    validateDir(validationdir, network);

    // Resize vectors in preparation for computation

    int confusionMatrixSize = validationTargets[0].size()+1;

    predictions.resize(confusionMatrixSize);
    actuals.resize(confusionMatrixSize);
    precisions.resize(confusionMatrixSize);
    recalls.resize(confusionMatrixSize);
    fones.resize(confusionMatrixSize);

    // Iterate through the validation targets and compute classifications
    for (int i = 0; i < validationTargets.size(); i++) {
        int target = validationTargets[0].size();
        float tempClassificationThreshold = classificationThreshold;
        for (int j = 0; j < validationTargets[i].size(); j++) {
            if (validationTargets[i][j] > tempClassificationThreshold) {
                target = j;
            }
        }
        targetClassifications.push_back(target);
        actuals[target]++;
    }

    int correct = 0;
    int wrong = 0;

    confusion.resize(confusionMatrixSize, std::vector<int>(confusionMatrixSize));

    // Iterate through the validation outputs and compute classifications & confusion matrix
    for (int i = 0; i < validationOutputs.size(); i++) {
        int classification = validationOutputs[0].size();
        float tempClassificationThreshold = classificationThreshold;
        std::cout << "Output: ";
        for (int j = 0; j < validationOutputs[i].size(); j++) {
            std::cout << validationOutputs[i][j] << " ";
            if (validationOutputs[i][j] > tempClassificationThreshold) {
                classification = j;
            }
        }
        std::cout << "\n";
        std::cout << "Classification: " << classification << "\n";
        std::cout << "Target: " << targetClassifications[i] << "\n";
        std::cout << "\n";

        outputClassifications.push_back(classification);

        confusion[classification][targetClassifications[i]] += 1;
        predictions[classification]++;

        if (classification == targetClassifications[i]) {
            correct++;
        } else {
            wrong++;
        }

    }

    // Print the confusion matrix
    std::cout << "Predicted | pu | su | lu | no \n";

    for (int i = 0; i < confusion[0].size(); i++) {
        std::cout << caLabels[i];
        for (int j = 0; j < confusion[0].size()-1; j++) {
            if (confusion[i][j] < 10) {
                std::cout << " "; //padding
            }
            std::cout << std::to_string(confusion[i][j]) + " | ";
        }
        std::cout << std::to_string(confusion[i][confusion[0].size()-1]) << "  \n";
    }
    std::cout << "\n";

    // compute and print precision, recall and F1 rates
    std::cout << "            Recall | Precision | F1\n";
    for (int i = 0; i < predictions.size(); i++) {
        if (actuals[i] > 0) {
            recalls[i] = confusion[i][i] / float(actuals[i]);
        } else {
            recalls[i] = 0;
        }
        uar += recalls[i];
        if (predictions[i] > 0) {
            precisions[i] = confusion[i][i] / float(predictions[i]);
        } else {
            precisions[i] = 0;
        }
        if (precisions[i] + recalls[i] > 0) {
            fones[i] = (2.0 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i]);
        } else {
            fones[i] = 0;
        }
        std::cout << caLabels[i] << " " << recalls[i];
        if (recalls[i] == 0) {
            std::cout << "  ";
        }
        std::cout << "   |   " << precisions[i];
        if (precisions[i] == 0) {
            std::cout << "  ";
        }
        std::cout <<  "   |   " << fones[i] << "\n";
    }
    std::cout <<"\n";

    std::cout << "Correct: " << correct << "\n";
    std::cout << "Wrong: " << wrong << "\n";
    uar = uar / predictions.size();
    std::cout << "Classification rate: " << 100 * correct/float(correct + wrong) << "%\n";
    std::cout << "Error rate: " << 100 * wrong/float(correct + wrong) << "%\n";
    std::cout << "Unweighted Average Recall: " << uar * 100 << "%\n";

    saveNetwork(config_file_location, network);
}


/*
 * Main training program for the network.
 *
 * Run from command line as follows:
 *
 * train config_filename dirname|log_filename [suffix]
 *
 * Must be run from the linux/ directory
 */

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <random>
#include <algorithm>

#include "../../network/src/network-linux.hpp"
#include "../../network/src/network-saveload-linux.hpp"
#include "training-set.hpp"

bool directory = false;

std::vector<std::vector<float>> trainingInputs;
std::vector<std::vector<float>> trainingTargets;
std::vector<std::vector<float>> trainingOutputs;

std::string suffix = "";

long examplesTrainedOn = 0;
float latestErrorRate = 0;

void loadTrainingSets(std::string filename, Network_L *network) {
    // Check the training file exists, and if it doesn't, exit
    std::ifstream check_logfile(filename);
    if (!check_logfile.good() || filename.find(suffix) == std::string::npos) {
        check_logfile.close();
        std::cout << filename << " is an invalid log file, skipping.\n";
    } else {

        TrainingSet *set = loadTrainingSet(filename);

        for (int i = 0; i < set->inputs.size(); i++) {
            trainingInputs.push_back(set->inputs[i]);
            trainingTargets.push_back(set->targets[i]);
        }
    }
}

void loadDir(std::string dirname, Network_L *network) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (dirname.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            if (ent->d_type == DT_REG &&
                std::string(ent->d_name).find(suffix) != std::string::npos) {
                loadTrainingSets(dirname + std::string(ent->d_name), network);
            }else if (ent->d_type == DT_DIR &&
                      std::string(ent->d_name).find(".") == std::string::npos) {
                loadDir(dirname + std::string(ent->d_name) + "/", network);
            }
        }
        closedir (dir);
    } else {
        std::cout << "Could not open directory " << dirname << "\n";
    }
}

int main(int argc, char * argv[]) {
    // Parse arguments
    if (argc < 3) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 4) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    if (argc == 4) {
        suffix = argv[3];
    }

    Network_L *network;

    // Check the network config file exists, and if it doesn't, create it.
    std::cout << "Checking network config file...";
    std::ifstream check_config(argv[1]);
    if (!check_config.is_open()) {
        std::cout << "not found, exiting\n";
        return 1;
    } else {
        std::cout << "found, loading network\n";
        // Load the network
        network = loadNetwork(argv[1]);
    }

    //determine if filename is a directory or not
    DIR *d;
    if ((d = opendir (argv[2])) != NULL) {
        directory = true;
    }

    // Load the given file(s)
    if (directory) {
        loadDir(argv[2], network);
    } else {
        loadTrainingSets(argv[2], network);
    }

    // Save for later
    float lr = network->getLearningRate();
    float m  = network->getMomentum();

    // Now train the network on these files
    // Create a vector of indexes and shuffle it
    std::vector<int> indexes(trainingInputs.size());
    std::iota (std::begin(indexes), std::end(indexes), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indexes.begin(), indexes.end(), g);

    // Now train the network in this random order
    int currentIndex;
    for (int i = 0; i < indexes.size(); i++) {
        currentIndex = indexes[i];
        latestErrorRate = network->trainNetwork(trainingInputs[currentIndex], trainingTargets[currentIndex]);
        examplesTrainedOn++;

        if (examplesTrainedOn % 100 == 0) {
            std::cout << "Trained " << examplesTrainedOn << " examples. Error rate is " << latestErrorRate << "\n";
        }
    }
    std::cout << "Finished training after " << examplesTrainedOn << " examples. Error rate is " << latestErrorRate << "\n";
    std::cout << "\n";

    // Restore original learning rate and momentum for inspection
    network->setLearningRate(lr);
    network->setMomentum(m);

    saveNetwork(argv[1], network);
}


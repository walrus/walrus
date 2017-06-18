/*
 * Main evaluation program for the network.
 *
 * Run from command line as follows:
 *
 * train config_filename [trainingdir] validationdir
 *
 * Will recursively scan directory  trainingdir and  read every log file with the _normalised suffix
 * and train the network on the data contained in it.
 *
 * If trainingdir argument is omitted, will not train
 *
 * Once done will validate the network on the contents of validationdir
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

std::string config_file_location =  "../network/config/network.txt";
bool training = false;
long examplesTrainedOn = 0;
float latestErrorRate = 0;
std::string trainingdir;
std::string validationdir;

std::vector<std::vector<float>> trainingInputs;
std::vector<std::vector<float>> trainingTargets;
std::vector<std::vector<float>> trainingOutputs;

std::vector<std::vector<float>> validationTargets;
std::vector<std::vector<float>> validationOutputs;

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

void loadTrainingSets(std::string filename, Network_L *network) {
    // Check the training file exists, and if it doesn't, exit
    std::ifstream check_logfile(filename);
    if (!check_logfile.good() || filename.find("_normalised") == std::string::npos) {
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
                std::string(ent->d_name).find("_normalised") != std::string::npos) {
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

    config_file_location = argv[1];
    if (argc == 4) {
        trainingdir = argv[2];
        validationdir = argv[3];
        training = true;
    } else {
        validationdir = argv[2];
    }


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

    // Train the network on the data in the given directory
    if (training) {
        loadDir(trainingdir, network);

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
    }


    std::cout << "Validating...\n";



    saveNetwork(config_file_location, network);
}


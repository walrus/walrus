/*
 * Main training program for the network.
 *
 * Run from command line as follows:
 *
 * train config_filename|-n filename|directory
 *
 * Will read every log file with the _normalised suffix and train the network on the data contained in it
 * The -n flag will use the network whose config is stored in network/config/network.txt on the data.
 *
 * Must be run from the linux/ directory
 */

#include <iostream>
#include <fstream>

#include "../../network/src/network.hpp"
#include "../../network/src/network-io.hpp"
#include "training-set.hpp"

std::string config_file_location =  "../network/config/network.txt";

void trainSet(std::string filename, Network *network) {
    TrainingSet *set = loadTrainingSet(filename);

    for (int i = 0; i < set->inputs.size(); i++) {
        network->trainNetwork(set->inputs[i], set->targets[i]);
    }
}

int main(int argc, char * argv[]) {
    // Parse arguments
    if (argc < 3) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 3) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    // If we're given a file location, use that instead of the default
    if (argv[1] != "-n") {
        config_file_location = argv[1];
    }

    Network *network;

    // Check the network config file exists, and if it doesn't, create it.
    std::cout << "Checking network config file...";
    std::ifstream check_config(config_file_location);
    if (!check_config.is_open()) {
        std::cout << "not found, creating new network\n";
        network = new Network(20, 10, 1, 0.3, 0.9, 0.5);
    } else {
        std::cout << "found, loading network\n";
        // Load the network
        network = loadNetwork(config_file_location);
    }

    std::cout << "Checking training file...";
    // Check the training file exists, and if it doesn't, exit
    std::ifstream check_logfile (argv[2]);
    if (!check_logfile.good() || std::string(argv[2]).find("_normalised") == std::string::npos) {
        check_logfile.close();
        std::cout << "invalid log file, exiting.\n";
        return 1;
    } else {
        std::cout << "ok\n";
    }

    std::cout << "Training\n";
    trainSet(argv[1], network);

    saveNetwork(config_file_location, network);

}


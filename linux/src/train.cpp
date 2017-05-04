/*
 * Main training program for the network.
 *
 * Run from command line as follows:
 *
 * train filename|directory
 *
 * Will read every log file with the _normalised suffix and train the network
 * whose config is stored in network/config/network.txt on the data.
 */

#include <iostream>
#include <fstream>

#include "../../network/src/network.hpp"
#include "../../network/src/network-io.hpp"
#include "training-set.hpp"

const std::string CONFIG_FILE_LOCATION =  "../../network/config/network.txt";

void trainSet(std::string filename, Network *network) {
    TrainingSet *set = loadTrainingSet(filename);

    for (int i = 0; i < set->inputs.size(); i++) {
        network->trainNetwork(set->inputs[i], set->targets[i]);
    }
}

int main(int argc, char * argv[]) {
    // Parse arguments
    if (argc < 2) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 2) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    // Check the network config file exists, and if it doesn't, create it.
    std::ifstream check_config (CONFIG_FILE_LOCATION);
    if (!check_config.good()) {
        std::cout << "Creating new network\n";
        Network *new_network = new Network(20, 10, 1, 0.3, 0.9, 0.5);
        saveNetwork(CONFIG_FILE_LOCATION, new_network);
    }

    // Load the network
    Network *network = loadNetwork(CONFIG_FILE_LOCATION);

    trainSet(argv[1], network);

    saveNetwork(CONFIG_FILE_LOCATION, network);
}


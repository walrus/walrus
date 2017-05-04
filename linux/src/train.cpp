/*
 * Main training program for the network.
 *
 * Run from command line as follows:
 *
 * train filename|directory
 *
 * Will read every log file with the _normalised suffix and train the network
 * whose config is stored in network/config/network.txt on the data.
 *
 * Must be run from the linux/ directory
 */

#include <iostream>
#include <fstream>

#include "../../network/src/network.hpp"
#include "../../network/src/network-io.hpp"
#include "training-set.hpp"

const std::string config_file_location =  "../network/config/network.txt";

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
    std::ifstream check_config (config_file_location);
    if (!check_config.is_open()) {
        std::cout << "Creating new network\n";
        Network *new_network = new Network(20, 10, 1, 0.3, 0.9, 0.5);
        std::cout << "About to save network\n";
        saveNetwork(config_file_location, new_network);
        std::cout << "Saved new network\n";
    }

    std::cout << "Loading network\n";
    // Load the network
    Network *network = loadNetwork(config_file_location);

    std::cout << "Checking training file\n";
    // Check the training file exists, and if it doesn't, exit
    std::ifstream check_logfile (argv[1]);
    if (!check_logfile.good()) {
        check_logfile.close();
        std::cout << "Invalid log file\n";
        return 1;
    }

    std::cout << "Training\n";
    trainSet(argv[1], network);

    saveNetwork(config_file_location, network);

}


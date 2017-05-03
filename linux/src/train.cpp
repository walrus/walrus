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

int main(int argc, char * agrv[]) {
    // Parse arguments
    if (argc < 2) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 2) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    // Check the network config file exists, and if it doesnt create it
    ifstream check_config ("../../network/config/network.txt");
    if (!check_config.good()) {
        std::cout << "Creating new network\n";
        Network new_network = new Network(20, 10, 1, 0.3, 0.9, 0.5);
        saveNetwork("../../network/config/network.txt", new_network);
    }

    // Load the network
    Network network = loadNetwork("../../network/config/network.txt");
}


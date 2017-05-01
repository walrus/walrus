#include "network-io.hpp"

/*
 * Functions for saving and loading network configurations to and from files.
 */

Network *loadNetwork(std::string filename) {
    return nullptr;
}


int saveNetwork(std::string filename, Network network) {
    std::ofstream config_file (filename);
    if (!config_file.is_open() || config_file.bad()) {
        return 1; // Error code
    }
    config_file << std::to_string(network.getNumInputNodes()) + "\n";
    config_file << std::to_string(network.getNumHiddenNodes()) + "\n";
    config_file << std::to_string(network.getNumOutputNodes()) + "\n";
    config_file << std::to_string(network.getLearningRate()) + "\n";
    config_file << std::to_string(network.getMomentum()) + "\n";
    config_file << std::to_string(network.getInitialWeightMax()) + "\n";

    config_file.close();
    return 0;
}

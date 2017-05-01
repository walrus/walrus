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
    // Save main config data
    config_file << std::to_string(network.getNumInputNodes()) + "\n";
    config_file << std::to_string(network.getNumHiddenNodes()) + "\n";
    config_file << std::to_string(network.getNumOutputNodes()) + "\n";
    config_file << std::to_string(network.getLearningRate()) + "\n";
    config_file << std::to_string(network.getMomentum()) + "\n";
    config_file << std::to_string(network.getInitialWeightMax()) + "\n";

    // Save hidden weights
    int nin_plus_one = network.getNumInputNodes() +1;
    int nhn = network.getNumHiddenNodes();
    vector<vector<float>> hiddenWeights = network.getHiddenWeights();

    for (int i = 0; i < nin_plus_one; i++) {
        for (int j = 0; j < nhn; j++) {
            config_file << std::to_string(hiddenWeights[i][j]) + "\n";
        }
    }

    // Save output weights
    int nhn_plus_one = network.getNumHiddenNodes() + 1;
    int non = network.getNumOutputNodes();
    vector<vector<float>> outputWeights = network.getOutputWeights();

    for (int i = 0; i < nhn_plus_one; i++) {
        for (int j = 0; j < non; j++) {
            config_file << std::to_string(outputWeights[i][j]) + "\n";
        }
    }

    config_file.close();
    return 0;
}

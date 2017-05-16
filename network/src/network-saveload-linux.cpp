#include "network-saveload-linux.hpp"

/*
 * Functions for saving and loading network configurations to and from files.
 */

Network *loadNetwork(std::string filename) {
    // Open the file and read it into a vector of lines
    std::ifstream config_file(filename.c_str());
    vector<std::string> lines;
    std::string line;

    while (std::getline(config_file, line))
    {
        lines.push_back(line);
    }

    // Parse the config data in the first six lines and create the network with the specified configuration
    int nin = std::stoi(lines[0]);
    int nhn = std::stoi(lines[1]);
    int non = std::stoi(lines[2]);

    float lr = std::stof(lines[3]);
    float m = std::stof(lines[4]);
    float iwm = std::stof(lines[5]);

    Network *network = new Network(nin, nhn, non, lr, m, iwm);

    // Parse the hidden weights
    int line_num = 6;
    vector<vector<float>> hiddenWeights;
    hiddenWeights.resize(nin+1, vector<float>(nhn));

    for (int i = 0; i < nin+1; i++) {
        for (int j = 0; j < nhn; j++) {
            hiddenWeights[i][j] = std::stof(lines[line_num]);
            line_num++;
        }
    }

    // Parse the output weights
    line_num = 6 + ((nin + 1) * nhn);
    vector<vector<float>> outputWeights;
    outputWeights.resize(nhn+1, vector<float>(non));

    for (int i = 0; i < nhn+1; i++) {
        for (int j = 0; j < non; j++) {
            outputWeights[i][j] = std::stof(lines[line_num]);
            line_num++;
        }
    }

    network->loadWeights(hiddenWeights, outputWeights);

    return network;
}


int saveNetwork(std::string filename, Network *network) {
    std::ofstream config_file (filename);
    if (!config_file.is_open() || config_file.bad()) {
        return 1; // Error code
    }
    // Save main config data
    config_file << std::to_string(network->getNumInputNodes()) + "\n";
    config_file << std::to_string(network->getNumHiddenNodes()) + "\n";
    config_file << std::to_string(network->getNumOutputNodes()) + "\n";
    config_file << std::to_string(network->getLearningRate()) + "\n";
    config_file << std::to_string(network->getMomentum()) + "\n";
    config_file << std::to_string(network->getInitialWeightMax()) + "\n";

    // Save hidden weights
    int nin_plus_one = network->getNumInputNodes() +1;
    int nhn = network->getNumHiddenNodes();
    vector<vector<float>> hiddenWeights = network->getHiddenWeights();

    for (int i = 0; i < nin_plus_one; i++) {
        for (int j = 0; j < nhn; j++) {
            config_file << std::to_string(hiddenWeights[i][j]) + "\n";
        }
    }

    // Save output weights
    int nhn_plus_one = network->getNumHiddenNodes() + 1;
    int non = network->getNumOutputNodes();
    vector<vector<float>> outputWeights = network->getOutputWeights();

    for (int i = 0; i < nhn_plus_one; i++) {
        for (int j = 0; j < non; j++) {
            config_file << std::to_string(outputWeights[i][j]) + "\n";
        }
    }

    config_file.close();
    return 0;
}

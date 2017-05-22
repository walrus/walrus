/*
 * Create a new network with the given arguments:
 *
 * new-network filename nin nhn non lr m diwm
 *
 * Where:
 *
 * filename is the path from project/linux/ to the desired file.
 * It is recommended to place the file under ../network/config
 * Will overwrite any existing network config
 *
 * nin = numInputNeurons
 * nhn = numHiddenNeurons
 * non = numOutputNeurons
 * lr  = learningRate
 * m   = momentum
 * diwm = defaultInitialWeightMax
 *
 * Must be run from the project/linux/ directory
 */


#include <iostream>
#include <fstream>
#include <dirent.h>

#include "../../network/src/network-linux.hpp"
#include "../../network/src/network-saveload-linux.hpp"

std::string config_file_path =  "";

int nin;
int nhn;
int non;

float lr;
float m;
float diwm;

int main(int argc, char * argv[]) {
    // Parse arguments
    if (argc < 8) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 8) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

    config_file_path = argv[1];
    nin = atoi(argv[2]);
    nhn = atoi(argv[3]);
    non = atoi(argv[4]);

    lr = atof(argv[5]);
    m = atof(argv[6]);
    diwm = atof(argv[7]);

    std::cout << "Checking for existing network config file...";
    std::ifstream check_config(config_file_path);
    if (!check_config.is_open()) {
        std::cout << "not found, creating new network.\n";
        Network_L *network = new Network_L(nin, nhn, non, lr, m, diwm);
        saveNetwork(config_file_path, network);
        std::cout << "Network created.\n";
    } else {
        check_config.close();
        std::cout << "found existing network, exiting.\n";
        return 1;
    }
}
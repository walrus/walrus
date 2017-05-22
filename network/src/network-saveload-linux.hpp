/*
 * Functions for saving and loading network configurations to and from files.
 * Also includes function to generate weights for initialisation.
 *
 * WILL NOT COMPILE ON ARDUINO
 */

#include <iostream>
#include <fstream>
#include <random>

#include "network-linux.hpp"

#ifndef PROJECT_NETWORK_IO_H
#define PROJECT_NETWORK_IO_H

Network_L *loadNetwork(std::string filename);
int saveNetwork(std::string filename, Network_L *network);

#endif //PROJECT_NETWORK_IO_H

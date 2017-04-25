#ifndef PROJECT_NETWORK_IO_H
#define PROJECT_NETWORK_IO_H

#include "../../network/src/network.hpp

Network *loadNetwork(std::string filename);
void saveNetwork(Network network, std::string filename);

#endif //PROJECT_NETWORK_IO_H

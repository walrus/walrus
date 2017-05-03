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

int main(int argc, char * agrv[]) {
    // Parse arguments
    if (argc < 2) {
        std::cout << "Too few arguments supplied\n";
        return 1;
    } else if (argc > 2) {
        std::cout << "Too many arguments supplied\n";
        return 1;
    }

}


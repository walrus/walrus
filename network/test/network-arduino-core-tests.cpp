#include "../../lib/catch.hpp"
#include "../src/network-arduino.hpp"

#include <iostream>

/* Main unit test file for the Arduino network code. */

TEST_CASE("The network can be successfully initialised and used to classify") {

    GIVEN("A valid #included network config and some training data") {

        float inputData[] = { 18208, 19993, 23514, 23514,
                              23662, 22080, 20603, 17170 };

        THEN("The network can be initialised") {
            Network_A *network = new Network_A();

            THEN("It can classify the input data") {
                float * output = network->classify(inputData);
            }
        }
    }
}
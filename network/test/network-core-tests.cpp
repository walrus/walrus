#include "../../lib/catch.hpp"
#include "../src/network.hpp"
/* Main unit test file for the network code. */

SCENARIO("The network can be initialised and configured correctly") {

    WHEN("The network is instantiated using the constructor") {
        std::random_device rd;
        std::mt19937 m_mt (rd());

        Network network = Network(m_mt);

        // This is necessary because Catch struggles to match against #defines for...reasons
        int nin  = NUM_INPUT_NODES;
        int nhn  = NUM_HIDDEN_NODES;
        int non  = NUM_OUTPUT_NODES;

        float dlr  = DEFAULT_LEARNING_RATE;
        float dm   = DEFAULT_MOMENTUM;
        float diwm = DEFAULT_INITIAL_WEIGHT_MAX;

        THEN("The default parameters are set properly") {
            REQUIRE(network.getNumInputNodes() == nin);
            REQUIRE(network.getNumHiddenNodes() == nhn);
            REQUIRE(network.getNumOutputNodes() == non);

            REQUIRE(network.getLearningRate() == dlr);
            REQUIRE(network.getMomentum() == dm);
            REQUIRE(network.getInitialWeightMax() == diwm);
        }
        THEN("The other variables are set properly") {
            REQUIRE(network.getTrainingCycle() == 0);
            REQUIRE(network.getRandomFloat() <= 1.0f);
            REQUIRE(network.getRandomFloat() >= -1.0f);
            REQUIRE(network.getErrorRate() == 1.0f);
            REQUIRE(network.getAccumulatedInput() == 0.0f);
        }
    }
}
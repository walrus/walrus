#include "../../lib/catch.hpp"
#include "../src/network.hpp"
/* Main unit test file for the network code. */

TEST_CASE("The core network functionality is all correct") {

    WHEN("The network is instantiated using the constructor") {
        std::random_device rd;
        std::mt19937 m_mt(rd());
        std::uniform_real_distribution<float> test_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
        Network network = Network(m_mt);

        // This is necessary because Catch struggles to match against #defines for...reasons
        int nin = NUM_INPUT_NODES;
        int nhn = NUM_HIDDEN_NODES;
        int non = NUM_OUTPUT_NODES;

        float dlr = DEFAULT_LEARNING_RATE;
        float dm = DEFAULT_MOMENTUM;
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
        GIVEN("An already instantiated network") {
            THEN("The learning rate can be adjusted") {
                REQUIRE(network.getLearningRate() == dlr);
                network.setLearningRate(0.5f);
                REQUIRE(network.getLearningRate() == 0.5f);
            }
            THEN("The momentum can be adjusted") {
                REQUIRE(network.getMomentum() == dm);
                network.setMomentum(0.5f);
                REQUIRE(network.getMomentum() == 0.5f);
            }
            THEN("The initial weight maximum can be adjusted") {
                REQUIRE(network.getInitialWeightMax() == diwm);
                network.setInitialWeightMax(0.9f);
                REQUIRE(network.getInitialWeightMax() == 0.9f);
            }

            THEN("It can (badly) attempt to classify without training") {
                vector<float>  input;

                for (int i = 0; i++; i < nin) {
                    input[i] = test_dist(m_mt);
                }

                //This is a bit messy, but will work until I replace all the arrays with vectors
                vector<float> output = network.classify(input);

                for (int i = 0; i++; i < non) {
                    REQUIRE(output[i] > -1.0f);
                    REQUIRE(output[i] < 1.0f);
                }
            }
            THEN("It can be trained") {
                vector<float> input;
                vector<float> output;

                for (int i = 0; i++; i < nin) {
                    input[i] = test_dist(m_mt);
                }

                for (int i = 0; i++; i < non) {
                    output[i] = test_dist(m_mt);
                }

                //This is a bit messy, but will work until I replace all the arrays with vectors
                float error = network.trainNetwork(input, output);

                REQUIRE(error > 0.0f);
            }
        }
    }
}
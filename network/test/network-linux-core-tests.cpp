#include "../../lib/catch.hpp"
#include "../src/network-linux.hpp"
/* Main unit test file for the network code. */

TEST_CASE("The core network functionality is all correct") {
    std::random_device rd;
    std::mt19937 m_mt(rd());
    std::uniform_real_distribution<float> test_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    std::uniform_real_distribution<float> target_dist = std::uniform_real_distribution<float>(0.0f, 1.0f);


    int nin = 8;
    int nhn = 7;
    int non = 4;

    float dlr = 0.3;
    float dm = 0.9;
    float diwm = 0.5;

    long tc = 0;

    GIVEN("The network is instantiated using the constructor") {
        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm, tc);

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
            REQUIRE(network.getErrorRate() == 0.0f);
            REQUIRE(network.getAccumulatedInput() == 0.0f);
        }
        THEN("The hidden weights are all in the correct range") {
            std::vector<std::vector<float>> hiddenWeights = network.getHiddenWeights();
            for (int i = 0; i < nin+1; i++) {
                for (int j = 0; j < nhn; j++) {
                    REQUIRE(hiddenWeights[i][j] < diwm);
                    REQUIRE(hiddenWeights[i][j] > -1 * diwm);
                }
            }
        }
        THEN("The output weights are all in the correct range") {
            std::vector<std::vector<float>> outputWeights = network.getOutputWeights();
            for (int i = 0; i < nhn+1; i++) {
                for (int j = 0; j < non; j++) {
                    REQUIRE(outputWeights[i][j] < diwm);
                    REQUIRE(outputWeights[i][j] > -1 * diwm);
                }
            }
        }
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
            std::vector<float>  input;
            input.resize(nin);

            for (int i = 0; i++; i < nin) {
                input[i] = test_dist(m_mt);
            }

            std::vector<float> output = network.classify(input);

            for (int i = 0; i++; i < non) {
                REQUIRE(output[i] > -1.0f);
                REQUIRE(output[i] < 1.0f);
            }
        }
        THEN("It can be trained") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float error = network.trainNetwork(input, target);

            REQUIRE(error > 0.0f);
        }
        THEN("Training reduces the error") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }

            float untrained_error = network.trainNetwork(input, target);

            REQUIRE(untrained_error > 0.0f);

            for (int i =0; i < 9; i++) {
                network.trainNetwork(input, target);
            }

            float trained_error = network.trainNetwork(input, target);

            REQUIRE(trained_error < untrained_error);
            REQUIRE(trained_error > 0.0f);
        }
    }

    GIVEN("Pre computed hidden and output weights") {
        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm, tc);

        // Generate hidden and output node weights to load
        std::vector<std::vector<float>> hiddenWeights;
        hiddenWeights.resize(nin+1, std::vector<float>(nhn));

        for (int i = 0; i < nhn; i++) {
            for (int j = 0; j <= nin; j++) {
                float randomFloat = test_dist(m_mt);
                hiddenWeights[j][i] = randomFloat * 0.5f;
            }
        }

        std::vector<std::vector<float>> outputWeights;
        outputWeights.resize(nhn+1, std::vector<float>(non));

        for(int i = 0 ; i < non ; i ++ ) {
            for(int j = 0 ; j <= nhn ; j++ ) {
                float randomFloat = test_dist(m_mt);
                outputWeights[j][i] = randomFloat * 0.5f ;
            }
        }
        THEN("The weights can be loaded") {
            std::vector<std::vector<float>> existingHiddenWeights = network.getHiddenWeights();
            std::vector<std::vector<float>> existingOutputWeights = network.getOutputWeights();

            network.loadWeights(hiddenWeights, outputWeights);

            std::vector<std::vector<float>> loadedHiddenWeights = network.getHiddenWeights();
            std::vector<std::vector<float>> loadedOutputWeights = network.getOutputWeights();

            REQUIRE(loadedHiddenWeights == hiddenWeights);
            REQUIRE(loadedOutputWeights == outputWeights);

            REQUIRE(loadedHiddenWeights != existingHiddenWeights);
            REQUIRE(loadedOutputWeights != existingOutputWeights);
        }
        THEN("It can attempt to classify without training") {
            std::vector<float>  input;
            input.resize(nin);

            for (int i = 0; i++; i < nin) {
                input[i] = test_dist(m_mt);
            }

            std::vector<float> output = network.classify(input);

            for (int i = 0; i++; i < non) {
                REQUIRE(output[i] > -1.0f);
                REQUIRE(output[i] < 1.0f);
            }
        }
        THEN("It can be trained") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float error = network.trainNetwork(input, target);

            REQUIRE(error > 0.0f);
        }
        THEN("Training reduces the error") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float untrained_error = network.trainNetwork(input, target);

            REQUIRE(untrained_error > 0.0f);

            for (int i =0; i < 9; i++) {
                network.trainNetwork(input, target);
            }

            float trained_error = network.trainNetwork(input, target);

            REQUIRE(trained_error < untrained_error);
            REQUIRE(trained_error > 0.0f);
        }
    }

    GIVEN("A network with very few neurons and edge case parameters") {

        nin = 2;
        nhn = 2;
        non = 1;
        dlr = 0.1;
        dm = 0.0;

        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm, tc);

        THEN("It can attempt to classify without training") {
            std::vector<float>  input;
            input.resize(nin);

            for (int i = 0; i++; i < nin) {
                input[i] = test_dist(m_mt);
            }

            std::vector<float> output = network.classify(input);

            for (int i = 0; i++; i < non) {
                REQUIRE(output[i] > -1.0f);
                REQUIRE(output[i] < 1.0f);
            }
        }
        THEN("It can be trained") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float error = network.trainNetwork(input, target);

            REQUIRE(error > 0.0f);
        }
        THEN("Training reduces the error") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float untrained_error = network.trainNetwork(input, target);

            REQUIRE(untrained_error > 0.0f);

            for (int i =0; i < 9; i++) {
                network.trainNetwork(input, target);
            }

            float trained_error = network.trainNetwork(input, target);

            REQUIRE(trained_error < untrained_error);
            REQUIRE(trained_error > 0.0f);
        }
    }

    GIVEN("A network using the ReLu activation function") {
        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm, tc);

        network.setActivationFunction(ActivationFunction::ReLu);

        THEN("The activation function is set properly") {
            REQUIRE(network.getActivationFunction() == ActivationFunction::ReLu);
        }
        THEN("It can be trained") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }


            float error = network.trainNetwork(input, target);

            REQUIRE(error > 0.0f);
        }
        THEN("Training reduces the error") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }

            float untrained_error = network.trainNetwork(input, target);

            REQUIRE(untrained_error > 0.0f);

            for (int i = 0; i < 9; i++) {
                network.trainNetwork(input, target);
            }

            float trained_error = network.trainNetwork(input, target);

            REQUIRE(trained_error < untrained_error);
            REQUIRE(trained_error > 0.0f);
        }
    }

    GIVEN("A network using the CrossEntropy error function") {
        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm, tc);

        network.setErrorFunction(ErrorFunction::CrossEntropy);

        THEN("The error function is set properly") {
            REQUIRE(network.getErrorFunction() == ErrorFunction::CrossEntropy);
        }
        THEN("It can be trained") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }

            float error = network.trainNetwork(input, target);

            REQUIRE(error > 0.0f);
        }
        THEN("Training reduces the error") {
            std::vector<float> input;
            input.resize(nin);
            std::vector<float> target;
            target.resize(non);

            for (int i = 0; i < nin; i++) {
                input[i] = test_dist(m_mt);
            }

            for (int i = 0; i < non; i++) {
                target[i] = target_dist(m_mt);
            }

            float untrained_error = network.trainNetwork(input, target);

            REQUIRE(untrained_error > 0.0f);

            for (int i =0; i < 9; i++) {
                network.trainNetwork(input, target);
            }

            float trained_error = network.trainNetwork(input, target);

            REQUIRE(trained_error < untrained_error);
            REQUIRE(trained_error > 0.0f);
        }
    }
}
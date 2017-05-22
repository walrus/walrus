/* Test functions for saving and loading network configurations to and from files. */

#include "../src/network-saveload-linux.hpp"
#include "../../lib/catch.hpp"

TEST_CASE("Network configurations can be saved to file and loaded from file") {
    GIVEN("A suitably configured network") {
        std::random_device rd;
        std::mt19937 m_mt(rd());
        std::uniform_real_distribution<float> test_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

        int nin = 8;
        int nhn = 7;
        int non = 4;

        float dlr = 0.3;
        float dm = 0.9;
        float diwm = 0.5;

        Network_L *network = new Network_L(nin, nhn, non, dlr, dm, diwm);

        std::string filename = "test_network_config.h";

        int status_code = saveNetwork(filename, network);

        THEN("It can be saved to file") {
            REQUIRE(status_code == 0);
        }

        // Open the file and read it into a vector of lines
        std::ifstream config_file(filename.c_str());
        std::vector<std::string> lines;
        std::string line;

        while (std::getline(config_file, line))
        {
            lines.push_back(line);
        }

        THEN("The main configuration options are all recorded") {
            REQUIRE(lines.size() >= 6);
        }

        THEN("The first line records the number of input nodes correctly") {
            std::string ninBody = "const int numInputNodes =";
            REQUIRE(lines[0].substr(0,25) == ninBody);
            int fileNin = std::stoi(lines[0].substr(26, lines[0].length() - 2));
            REQUIRE(fileNin == nin);
        }

        THEN("The second line records the number of hidden nodes correctly") {
            std::string nhnBody = "const int numHiddenNodes =";
            REQUIRE(lines[1].substr(0,26) == nhnBody);
            int fileNhn = std::stoi(lines[1].substr(27, lines[1].length() - 2));
            REQUIRE(fileNhn == nhn);
        }

        THEN("The third line records the number of output nodes correctly") {
            std::string nonBody = "const int numOutputNodes =";
            REQUIRE(lines[2].substr(0,26) == nonBody);
            int fileNon = std::stoi(lines[2].substr(27, lines[2].length() - 2));
            REQUIRE(fileNon == non);
        }

        THEN("The fourth line records the learning rate correctly") {
            std::string lrBody = "const float learningRate =";
            REQUIRE(lines[3].substr(0, 26) == lrBody);
            float fileLr = std::stof(lines[3].substr(27, lines[3].length()-28));
            REQUIRE(fileLr == dlr);
        }

        THEN("The fifth line records the momentum correctly") {
            std::string mBody = "const float momentum =";
            REQUIRE(lines[4].substr(0, 22) == mBody);
            float fileM = std::stof(lines[4].substr(23, lines[4].length()-24));
            REQUIRE(fileM == dm);
        }

        THEN("The sixth line records the initial weight max correctly") {
            std::string iwmBody = "const float initialWeightMax =";
            REQUIRE(lines[5].substr(0, 30) == iwmBody);
            float fileIwm = std::stof(lines[5].substr(31, lines[5].length()-32));
            REQUIRE(fileIwm == diwm);
        }

        THEN("The seventh line is blank") {
            REQUIRE(lines[6] == "");
        }

        THEN("The eighth line records the hidden weight declaration") {
            REQUIRE(lines[7] == "float hiddenWeights[numInputNodes +1][numHiddenNodes] PROGMEM = {");
        }

        THEN("The hidden weights are all recorded") {
            REQUIRE(lines.size() >= 8 + (nin + 1));
        }

        THEN("The correct lines record the hidden weights correctly") {
            int line_num = 8;
            float weight_from_file, weight_from_vector;

            std::vector<std::vector<float>> hiddenWeights = network->getHiddenWeights();
            for (int i = 0; i < nin+1; i++) {
                std::string line = lines[line_num].substr(5, lines[line_num].length()- 8);
                float lineWeights[nhn];
                std::string value;
                std::istringstream iss(line);
                int k = 0;

                while (std::getline(iss, value, ',')) {
                    lineWeights[k] = stof(value);
                    k++;
                }

                for (int j = 0; j < nhn; j++) {
                    weight_from_file = lineWeights[j];
                    weight_from_vector = hiddenWeights[i][j];

                    REQUIRE(weight_from_file == Approx(weight_from_vector));
                }
                line_num++;
            }
        }

        int previous_lines = 8 + (nin + 1);

        THEN("The line after the hidden weights encloses the array") {
            REQUIRE(lines[previous_lines] == "};");
        }

        previous_lines++;

        THEN("The line after the array is blank") {
            REQUIRE(lines[previous_lines] == "");
        }

        previous_lines++;

        THEN("The next line records the output weight declaration") {
            REQUIRE(lines[previous_lines] == "float outputWeights[numHiddenNodes +1][numOutputNodes] PROGMEM = {");
        }

        previous_lines++;

        THEN("The output weights are all recorded") {
            REQUIRE(lines.size() >= previous_lines + (nhn + 1));
        }

        THEN("The correct lines record the output weights correctly") {
            int line_num = previous_lines;
            float weight_from_file, weight_from_vector;

            std::vector<std::vector<float>> outputWeights = network->getOutputWeights();
            for (int i = 0; i < nhn+1; i++) {
                std::string line = lines[line_num].substr(5, lines[line_num].length()- 8);
                float lineWeights[non];
                std::string value;
                std::istringstream iss(line);
                int k = 0;

                while (std::getline(iss, value, ',')) {
                    lineWeights[k] = stof(value);
                    k++;
                }

                for (int j = 0; j < non; j++) {
                    weight_from_file = lineWeights[j];
                    weight_from_vector = outputWeights[i][j];

                    REQUIRE(weight_from_file == Approx(weight_from_vector));
                }
                line_num++;
            }
        }

        previous_lines += nhn + 1;

        THEN("The line after the hidden weights encloses the array") {
            REQUIRE(lines[previous_lines] == "};");
        }

        previous_lines++;

        THEN("There is nothing else after the output weights") {
            REQUIRE(lines.size() == previous_lines);
        }

        config_file.close();

        GIVEN("A saved network") {

            Network_L loaded_network = *loadNetwork(filename);

            THEN("The network is created with the correct number of input nodes") {
                REQUIRE(loaded_network.getNumInputNodes() == nin);
            }

            THEN("The network is created with the correct number of hidden nodes") {
                REQUIRE(loaded_network.getNumHiddenNodes() == nhn);
            }

            THEN("The network is created with the correct number of output nodes") {
                REQUIRE(loaded_network.getNumOutputNodes() == non);
            }

            THEN("The network is created with the correct learning rate") {
                REQUIRE(loaded_network.getLearningRate() == Approx(dlr));
            }

            THEN("The network is created with the correct momentum") {
                REQUIRE(loaded_network.getMomentum() == Approx(dm));
            }

            THEN("The network is created with the correct initial weight max") {
                REQUIRE(loaded_network.getInitialWeightMax() == Approx(diwm));
            }

            THEN("The network is created with the correct hidden weights") {
                std::vector<std::vector<float>> savedHiddenWeights = network->getHiddenWeights();
                std::vector<std::vector<float>> loadedHiddenWeights = loaded_network.getHiddenWeights();

                for (int i = 0; i < nin+1; i++) {
                    for (int j = 0; j < nhn; j++) {
                        REQUIRE(loadedHiddenWeights[i][j] == Approx(savedHiddenWeights[i][j]));
                    }
                }
            }

            THEN("The network is created with the correct output weights") {
                std::vector<std::vector<float>> savedOutputWeights = network->getOutputWeights();
                std::vector<std::vector<float>> loadedOutputWeights = loaded_network.getOutputWeights();

                for (int i = 0; i < nhn+1; i++) {
                    for (int j = 0; j < non; j++) {
                        REQUIRE(loadedOutputWeights[i][j] == Approx(savedOutputWeights[i][j]));
                    }
                }
            }
        }
    }
}
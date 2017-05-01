 /* Test functions for saving and loading network configurations to and from files. */

#include "../src/network-io.hpp"
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

         Network network = Network(nin, nhn, non, dlr, dm, diwm);

         std::string filename = "test_config_file.txt";

         int status_code = saveNetwork(filename, network);

         THEN("It can be saved to file") {
             REQUIRE(status_code == 0);
         }

         // Open the file and read it into a vector of lines
         std::ifstream config_file(filename.c_str());
         vector<std::string> lines;
         std::string line;

         while (std::getline(config_file, line))
         {
             lines.push_back(line);
         }

         THEN("The main configuration options are all recorded") {
             REQUIRE(lines.size() >= 6);
         }

         THEN("The first line records the number of input nodes correctly") {
             REQUIRE(std::stoi(lines[0]) == nin);
         }

         THEN("The second line records the number of hidden nodes correctly") {
             REQUIRE(std::stoi(lines[1]) == nhn);
         }

         THEN("The third line records the number of output nodes correctly") {
             REQUIRE(std::stoi(lines[2]) == non);
         }

         THEN("The fourth line records the learning rate correctly") {
             REQUIRE(std::stof(lines[3]) == dlr);
         }

         THEN("The fifth line records the momentum correctly") {
             REQUIRE(std::stof(lines[4]) == dm);
         }

         THEN("The sixth line records the initial weight max correctly") {
             REQUIRE(std::stof(lines[5]) == diwm);
         }

         THEN("The hidden weights are all recorded") {
             REQUIRE(lines.size() >= 6 + ((nin + 1) * nhn));
         }

         THEN("The correct lines record the hidden weights correctly") {
             int line_num = 6;
             float weight_from_file, weight_from_vector;

             vector<vector<float>> hiddenWeights = network.getHiddenWeights();
             for (int i = 0; i < nin+1; i++) {
                 for (int j = 0; j < nhn; j++) {
                     weight_from_file = std::stof(lines[line_num]);
                     weight_from_vector = hiddenWeights[i][j];

                     REQUIRE(weight_from_file == Approx(weight_from_vector));
                     line_num++;
                 }
             }
         }

         int previous_lines = 6 + ((nin + 1) * nhn);

         THEN("The output weights are all recorded") {
             REQUIRE(lines.size() >= previous_lines + ((nhn + 1) * non));
         }

         THEN("The correct lines record the output weights correctly") {
             int line_num = previous_lines;
             float weight_from_file, weight_from_vector;

             vector<vector<float>> outputWeights = network.getOutputWeights();
             for (int i = 0; i < nhn+1; i++) {
                 for (int j = 0; j < non; j++) {
                     weight_from_file = std::stof(lines[line_num]);
                     weight_from_vector = outputWeights[i][j];

                     REQUIRE(weight_from_file == Approx(weight_from_vector));
                     line_num++;
                 }
             }
         }

         config_file.close();

         GIVEN("A saved network") {

             Network loaded_network = *loadNetwork(filename);

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
                 vector<vector<float>> savedHiddenWeights = network.getHiddenWeights();
                 vector<vector<float>> loadedHiddenWeights = loaded_network.getHiddenWeights();

                 for (int i = 0; i < nin+1; i++) {
                     for (int j = 0; j < nhn; j++) {
                         REQUIRE(loadedHiddenWeights[i][j] == Approx(savedHiddenWeights[i][j]));
                     }
                 }
             }

             THEN("The network is created with the correct output weights") {
                 vector<vector<float>> savedOutputWeights = network.getOutputWeights();
                 vector<vector<float>> loadedOutputWeights = loaded_network.getOutputWeights();

                 for (int i = 0; i < nhn+1; i++) {
                     for (int j = 0; j < non; j++) {
                         REQUIRE(loadedOutputWeights[i][j] == Approx(savedOutputWeights[i][j]));
                     }
                 }
             }
         }
     }
 }
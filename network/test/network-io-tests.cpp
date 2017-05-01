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
     }
 }
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

         std::string filename = "filename.txt";

         THEN("It can be saved to file") {
             int status_code = saveNetwork(filename, network);

             REQUIRE(status_code == 0);
         }
     }
 }
#include "../../lib/catch.hpp"
#include "../src/network-linux.hpp"
/* Legacy unit test file for the network code, to check that it can still do what the original code did */

TEST_CASE("The library can implement the original ArduinoANN code's functionality") {
    std::random_device rd;
    std::mt19937 m_mt(rd());
    std::uniform_int_distribution<int> pattern_dist = std::uniform_int_distribution<int>(0, 9);


    int nin = 8;
    int nhn = 7;
    int non = 4;

    float dlr = 0.3;
    float dm = 0.9;
    float diwm = 0.5;

    int PatternCount = 10;

    float Error = 1.0;
    float Success = 0.0004;

    int p, q, r;

    std::vector<std::vector<float>> Input = {
            { 1, 1, 1, 1, 1, 1, 0 },  // 0
            { 0, 1, 1, 0, 0, 0, 0 },  // 1
            { 1, 1, 0, 1, 1, 0, 1 },  // 2
            { 1, 1, 1, 1, 0, 0, 1 },  // 3
            { 0, 1, 1, 0, 0, 1, 1 },  // 4
            { 1, 0, 1, 1, 0, 1, 1 },  // 5
            { 0, 0, 1, 1, 1, 1, 1 },  // 6
            { 1, 1, 1, 0, 0, 0, 0 },  // 7
            { 1, 1, 1, 1, 1, 1, 1 },  // 8
            { 1, 1, 1, 0, 0, 1, 1 }   // 9
    };

    std::vector<std::vector<float>> Target = {
            { 0, 0, 0, 0 },
            { 0, 0, 0, 1 },
            { 0, 0, 1, 0 },
            { 0, 0, 1, 1 },
            { 0, 1, 0, 0 },
            { 0, 1, 0, 1 },
            { 0, 1, 1, 0 },
            { 0, 1, 1, 1 },
            { 1, 0, 0, 0 },
            { 1, 0, 0, 1 }
    };

    std::vector<int> indexes;
    indexes.reserve(PatternCount);
    for (int i = 0; i < PatternCount; ++i) {
        indexes.push_back(i);
    }

    GIVEN("A network with the example parameters") {
        Network_L network = Network_L(nin, nhn, non, dlr, dm, diwm);

        THEN("It will eventually succeed at training") {
            long TrainingCycle = 1;

            while(TrainingCycle < 2147483647 && Error > Success) {
                std::random_shuffle(indexes.begin(), indexes.end());

                for (q = 0; q < PatternCount; q++) {
                    p = indexes[q];
                    Error = network.trainNetwork(Input[p], Target[p]);
                }
                TrainingCycle++;
            }
            REQUIRE(Error < Success);
            REQUIRE(Error > 0.0f);
        }
    }
}
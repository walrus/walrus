#include "../../lib/catch.hpp"
#include "../src/training-set.hpp"

/* Main unit test file for the training code */

TEST_CASE("Loading training sets from files works correctly") {

    GIVEN("A correctly formatted, normalised log file") {

        std::string filename = "test_normalised_log_file.txt";

        // Create the file and populate it with a set of three repetitions
        std::ofstream log_file (filename);

        // First repetition (positive)
        log_file << "Repetition start\n";
        log_file << "17143\n";
        log_file << "15082\n";
        log_file << "13075\n";
        log_file << "22519\n";
        log_file << "23199\n";
        log_file << "25752\n";
        log_file << "24906\n";
        log_file << "23515\n";
        log_file << "22540\n";
        log_file << "21055\n";
        log_file << "19321\n";
        log_file << "16712\n";
        log_file << "15010\n";
        log_file << "14829\n";
        log_file << "17315\n";
        log_file << "16071\n";
        log_file << "15945\n";
        log_file << "16517\n";
        log_file << "17596\n";
        log_file << "17332\n";
        log_file << "Repetition end\n";
        log_file << "1\n"; // Target for 'press up' node

        // Second repetition (also positive)

        log_file << "Repetition start\n";
        log_file << "16792\n";
        log_file << "18305\n";
        log_file << "18953\n";
        log_file << "23043\n";
        log_file << "20958\n";
        log_file << "21154\n";
        log_file << "21154\n";
        log_file << "21062\n";
        log_file << "19921\n";
        log_file << "21268\n";
        log_file << "22366\n";
        log_file << "21201\n";
        log_file << "21256\n";
        log_file << "20413\n";
        log_file << "19228\n";
        log_file << "17263\n";
        log_file << "17596\n";
        log_file << "15419\n";
        log_file << "14801\n";
        log_file << "16446\n";
        log_file << "Repetition end\n";
        log_file << "1\n";

        // Third repetition (negative)
        log_file << "Repetition start\n";
        log_file << "20319\n";
        log_file << "23010\n";
        log_file << "22475\n";
        log_file << "24981\n";
        log_file << "22182\n";
        log_file << "23919\n";
        log_file << "23161\n";
        log_file << "20776\n";
        log_file << "20776\n";
        log_file << "20776\n";
        log_file << "20916\n";
        log_file << "23099\n";
        log_file << "21262\n";
        log_file << "21554\n";
        log_file << "21463\n";
        log_file << "21270\n";
        log_file << "22526\n";
        log_file << "21829\n";
        log_file << "22082\n";
        log_file << "22099\n";
        log_file << "Repetition end\n";
        log_file << "0\n";

        log_file.close();

        THEN("The file can be read") {
            REQUIRE_NOTHROW(loadTrainingSet(filename));
        }

        TrainingSet set = *loadTrainingSet(filename);

        THEN("All the inputs are recorded") {
            REQUIRE(set.inputs.size() == 3);
            REQUIRE(set.inputs[0].size() == 20);
            REQUIRE(set.inputs[1].size() == 20);
            REQUIRE(set.inputs[2].size() == 20);
        }

        THEN("All the targets are recorded") {
            REQUIRE(set.targets.size() == 3);
            REQUIRE(set.targets[0].size() == 1);
            REQUIRE(set.targets[1].size() == 1);
            REQUIRE(set.targets[2].size() == 1);
        }
    }
}
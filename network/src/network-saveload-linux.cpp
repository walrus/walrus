#include <sstream>

#include "network-saveload-linux.hpp"

/*
 * Functions for saving and loading network configurations to and from files.
 */

Network_L *loadNetwork(std::string filename) {
    // Open the file and read it into a vector of lines
    std::ifstream config_file(filename.c_str());
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(config_file, line))
    {
        lines.push_back(line);
    }

    // Parse the basic config data and create the network with the specified configuration
    int nin = std::stoi(lines[5].substr(26, lines[5].length() - 2));
    int nhn = std::stoi(lines[6].substr(27, lines[6].length() - 2));
    int non = std::stoi(lines[7].substr(27, lines[7].length() - 2));

    float lr = std::stof(lines[8].substr(27, lines[8].length()-28));
    float m = std::stof(lines[9].substr(23, lines[9].length()-24));
    float iwm = std::stof(lines[10].substr(31, lines[10].length()-32));

    long tc = std::stol(lines[12].substr(42, lines[12].length()-42));

    ActivationFunction haf = stringToAF(lines[13].substr(53, lines[13].length()-53));
    ActivationFunction oaf = stringToAF(lines[14].substr(53, lines[14].length()-53));

    ErrorFunction ef = stringToEF(lines[15].substr(42, lines[15].length()-42));

    Network_L *network = new Network_L(nin, nhn, non, lr, m, iwm, tc);

    network->setHiddenActivationFunction(haf);
    network->setOutputActivationFunction(oaf);
    network->setErrorFunction(ef);

    // Parse the hidden weights
    int line_num = 18;
    std::vector<std::vector<float>> hiddenWeights;
    hiddenWeights.resize(nin+1, std::vector<float>(nhn));

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
            hiddenWeights[i][j] = lineWeights[j];
        }
        line_num++;
    }

    // Parse the output weights
    line_num = 21 + (nin + 1);
    std::vector<std::vector<float>> outputWeights;
    outputWeights.resize(nhn+1, std::vector<float>(non));

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
            outputWeights[i][j] = lineWeights[j];
        }
        line_num++;
    }

    network->loadWeights(hiddenWeights, outputWeights);

    return network;
}

int saveNetwork(std::string filename, Network_L *network) {
    std::ofstream config_file (filename);
    if (!config_file.is_open() || config_file.bad()) {
        return 1; // Error code
    }
    // Save #define
    config_file << "#ifndef ARDUINO_CONFIG_H\n";
    config_file << "#define ARDUINO_CONFIG_H\n";
    config_file << "\n";

    // Save the #include so that PROGMEM works
    config_file << "#include \"avr/pgmspace.h\"\n";
    config_file << "\n";

    // Save main config data
    config_file << "const int numInputNodes = " << std::to_string(network->getNumInputNodes()) + ";\n";
    config_file << "const int numHiddenNodes = " <<  std::to_string(network->getNumHiddenNodes()) + ";\n";
    config_file << "const int numOutputNodes = " <<  std::to_string(network->getNumOutputNodes()) + ";\n";
    config_file << "const float learningRate = " <<  std::to_string(network->getLearningRate()) + ";\n";
    config_file << "const float momentum = " <<  std::to_string(network->getMomentum()) + ";\n";
    config_file << "const float initialWeightMax = " <<  std::to_string(network->getInitialWeightMax()) + ";\n";
    config_file << "\n";

    config_file << "// TrainingCycle (not needed on Arduino): " << std::to_string(network->getTrainingCycle()) <<"\n";
    config_file << "// hiddenActivationFunction (not needed on Arduino): " << aFToString(network->getHiddenActivationFunction()) <<"\n";
    config_file << "// outputActivationFunction (not needed on Arduino): " << aFToString(network->getOutputActivationFunction()) <<"\n";
    config_file << "// ErrorFunction (not needed on Arduino): " << eFToString(network->getErrorFunction()) <<"\n";

    config_file << "\n";


    // Save hidden weights
    int nin_plus_one = network->getNumInputNodes() +1;
    int nhn = network->getNumHiddenNodes();
    std::vector<std::vector<float>> hiddenWeights = network->getHiddenWeights();

    config_file << "const float hiddenWeights[numInputNodes +1][numHiddenNodes] PROGMEM = {\n";
    for (int i = 0; i < nin_plus_one; i++) {
        config_file << "    { ";
        for (int j = 0; j < nhn-1; j++) {
            config_file << std::to_string(hiddenWeights[i][j]) + ", ";
        }
        config_file << std::to_string(hiddenWeights[i][nhn-1]) << " }, \n";
    }
    config_file << "};\n";
    config_file << "\n";

    // Save output weights
    int nhn_plus_one = network->getNumHiddenNodes() + 1;
    int non = network->getNumOutputNodes();
    std::vector<std::vector<float>> outputWeights = network->getOutputWeights();

    config_file << "const float outputWeights[numHiddenNodes +1][numOutputNodes] PROGMEM = {\n";
    for (int i = 0; i < nhn_plus_one; i++) {
        config_file << "    { ";
        for (int j = 0; j < non-1; j++) {
            config_file << std::to_string(outputWeights[i][j]) + ", ";
        }
        config_file << std::to_string(outputWeights[i][non-1]) << " }, \n";
    }
    config_file << "};\n";
    config_file << "\n";
    config_file << "#endif // ARDUINO_CONFIG_H";

    config_file.close();
    return 0;
}

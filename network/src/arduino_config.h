#ifndef ARDUINO_CONFIG_H
#define ARDUINO_CONFIG_H

const int numInputNodes = 8;
const int numHiddenNodes = 7;
const int numOutputNodes = 4;
const float learningRate = 0.300000;
const float momentum = 0.900000;
const float initialWeightMax = 0.500000;

// NOTE THAT THIS DOES NOT INCLUDE PROGMEM - STRICTLY FOR RUNNING UNIT TESTS

const float hiddenWeights[numInputNodes +1][numHiddenNodes] = {
    { 0.437861, 0.203009, -0.464994, -0.259296, 0.495690, 0.193953, 0.223447 }, 
    { -0.236467, 0.396916, 0.059815, -0.023111, -0.488556, -0.448045, 0.072503 }, 
    { -0.081328, 0.332207, -0.442256, -0.216266, -0.025166, -0.429176, -0.322100 }, 
    { 0.469696, -0.028402, 0.149298, 0.317331, 0.059276, 0.407455, 0.217819 }, 
    { -0.412486, 0.397227, -0.232766, 0.453047, -0.485316, 0.424805, -0.441376 }, 
    { -0.411115, -0.053089, 0.285984, 0.428740, -0.419451, 0.486221, 0.390780 }, 
    { 0.246134, 0.340920, -0.382968, 0.184555, 0.485498, 0.341787, 0.174443 }, 
    { 0.445877, -0.330478, 0.433466, -0.346288, 0.379657, -0.160876, 0.072326 }, 
    { -0.395366, -0.218795, -0.042314, -0.284750, 0.098627, 0.465455, 0.133732 }, 
};

const float outputWeights[numHiddenNodes +1][numOutputNodes] = {
    { 0.278773, 0.348158, -0.350823, -0.221428 }, 
    { 0.206889, -0.426063, 0.065177, 0.142669 }, 
    { 0.013450, -0.176797, -0.137473, -0.497551 }, 
    { -0.363868, 0.138731, -0.129507, -0.172391 }, 
    { 0.313267, 0.269420, -0.318927, -0.202453 }, 
    { 0.171745, -0.398208, -0.385978, -0.117924 }, 
    { -0.263365, -0.127395, -0.479104, -0.119527 }, 
    { 0.445885, 0.496179, 0.049553, -0.036793 }, 
};

#endif // ARDUINO_CONFIG_H
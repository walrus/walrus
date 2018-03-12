# WALRUS
Walrus is a feed forward neural network library for Arduino devices. It allows a network to be trained on a Linux machine and then uploaded to an Arduino for predictions. 

## Why walrus?

[This is why](https://imgur.com/gallery/GUnt3yw)

## Dependencies

You will need an Arduino board with enough RAM to fit the network into memory - the more the better.

- [Arduino IDE (v1.6 or later)](https://www.arduino.cc/en/Main/Software)
- [Curie IMU (library)](https://www.arduino.cc/en/Reference/CurieIMU)
- [ArduinoANN](http://robotics.hobbizine.com/arduinoann.html)
- [Python](https://www.python.org/downloads/)
- [GCC 6](https://gcc.gnu.org/gcc-6/)
- [Catch](https://github.com/philsquared/Catch)

## Setting up git hooks

To install version-controlled git hooks, run the following bash script:

    ./.init-hooks

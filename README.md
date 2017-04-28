# WALRUS
Exercise classification using an Artificial Neural Network running on an Intel Curie/Genuino 101, for my BEng Individual Project.

## Why walrus?

[This is why](https://imgur.com/gallery/GUnt3yw)

## Dependencies

An Intel Curie.
Bluetooth Low Energy (BLE) capable Android phone

- [Arduino IDE](https://www.arduino.cc/en/Main/Software)
- [Curie IMU (library)](https://www.arduino.cc/en/Reference/CurieIMU)
- [ArduinoANN](http://robotics.hobbizine.com/arduinoann.zip)
- [Python 2](https://www.python.org/downloads/) (Hoping to switch to Python 3 soon...)
- [GCC 6](https://gcc.gnu.org/gcc-6/)
- [Catch](https://github.com/philsquared/Catch)

Plus more to come, undoubtedly.

## Setting up git hooks

To install version-controlled git hooks, run the following script:

    ./.init-hooks

## Training the Classifier

# Logging data from the Curie

To log data from the curie, upload the sketch `logger.ino`, then run `./.log-data` 

# Normalising data for training

The normalisation script has multiple options. Once you've logged enough data, either run `./.normalise-data file` for each file, or run `./.normalise-data -r directory` to normalise all the files in the given directory.
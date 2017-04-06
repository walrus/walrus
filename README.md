# WALRUS
Exercise classification using Numenta's Hierarchical Temporal Memory (HTM) on an Intel Curie, for my BEng Individual Project.

## Why walrus?

[This is why](https://imgur.com/gallery/GUnt3yw)

## Dependencies

- [Python 2.7](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/installing/)>=8.1.2
- [Nupic](https://github.com/numenta/nupic)
    - [setuptools](https://setuptools.readthedocs.io)>=25.2.0
    - [wheel](http://pythonwheels.com)>=0.29.0
    - [numpy](http://www.numpy.org/)
    - C++ 11 compiler like [gcc](https://gcc.gnu.org/) (4.8+) or [clang](http://clang.llvm.org/)
- [Nupic Core](https://github.com/numenta/nupic.core)
    - [pycapnp](https://jparyani.github.io/pycapnp/)==0.5.8
    - [CMake](https://cmake.org/)

## Setting up git hooks

To install version-controlled git hooks, run the following script:

    ./.init-hooks

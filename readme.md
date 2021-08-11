# CMake Docker GitHub Actions

This repository is for testing the use of Github Actions to create python wheels inside manylinux containers, via CMake.

Creating a container image itself might make more sense in general, but this is testing for making binary python wheels for [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2).

The goals are:

+ CMake project producing
  + A CUDA C++ static library (initially C++ only)
  + One or more compiled executables
  + A mock test suite
  + Python3 bindings via swig, stored in binary wheels.
+ Github Actions which:
  + Build targets using separate steps on branch pushes / PRs?
  + Perform a more thorough build on pushes to tags, for a few key matrix entries
  + Build an assortment of python wheels, with a larger build matrix (once it works), which conform to the manylinux2014 standard.

## CI Jobs

> @todo Talk about the CI jobs?

## Building Locally

### Requirements

+ CMake >= 3.18
+ C++17 compiler, such as `gcc 9`, or Visual Studio 2019
+ `Swig 4`

### Linux

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_SWIG_PYTHON=ON
cmake --build . --target all -j `nproc`
```

### Windows

> @todo - see if windows works? Should just need -A x64 and -G "Visual Studio 16 2019", although maybe not if CMake is new enough. Then also use --config Release and --target ALL_BUILD

## Running "Tests"

To run the c++ "test suite" for the `Release` config:

```bash
./bin/Release/tests
```

To run the c++ python "test suite" for the `Release` config:

```bash
source lib/Release/python/venv/bin/activate
python3 ../tests/swig/python/tests.py
```

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

#!/usr/bin/python

# noinspection PyUnresolvedReferences,PyUnresolvedReferences

import os, sys, subprocess

# Flags:
#
# -n means recompile network (for use if the network code has changed)
#
# This script should be run from linux/


n_flag = False          # Whether or not the -n 'recompile network' flag is set


def run_tests():
    # Compile core tests
    print("Compiling tests...")
    a = subprocess.Popen(["g++", "-c", "-std=c++11", "test/training-io-tests.cpp"])
    a.wait()
    if a.returncode == 1:
        sys.exit(1) 

    # Link the various bits together into an executable
    print("Linking...")
    b = subprocess.Popen(["g++", "../catch-main.o", "training-io-tests.o", "training-set.o", "../network/network-linux.o", "-o", ".catch.exe", "-std=c++11"])
    b.wait()
    if b.returncode == 1:
        sys.exit(1)

    print("Running tests...")
    # Run the tests
    c = subprocess.Popen(["./.catch.exe"])
    c.wait()
    if c.returncode == 1:
        sys.exit(1)


#
# Main Program
#


# Check for being in linux/
_, cwd = os.path.split(os.getcwd())
if not cwd == "linux":
    print("Please run from the project/linux/ folder, not %s/" % cwd)
    sys.exit(1)


# Parse arguments
# noinspection PyUnresolvedReferences
if len(sys.argv) == 2:
    if sys.argv[1] == "-n":
        n_flag = True
    else:
        print("%s is not a valid argument; try again." % sys.argv[1])
        sys.exit(1)
# noinspection PyUnresolvedReferences
if len(sys.argv) > 2:
    print("Too many arguments given; try again.")
    sys.exit(1)


# If the main test object file doesn't exist, compile it
if not (os.path.isfile("../catch-main.o")):
    print("Compiling main...")
    m = subprocess.Popen(["g++", "-c", "-std=c++11", "../catch-main.cpp", "-o", "../catch-main.o"])
    m.wait()
    if m.returncode == 1:
        sys.exit(1) 


# If the network code doesn't exist, or the -n flag is set, compile it
if not (os.path.isfile("../network/network-linux.o")) or n_flag:
    print("Compiling network")
    n = subprocess.Popen(["g++", "-c", "-std=c++11", "../network/src/network-linux.cpp"])
    n.wait()
    if n.returncode == 1:
            sys.exit(1)  

# Compile training code
print("Compiling training code")
t = subprocess.Popen(["g++", "-c", "-std=c++11", "src/training-set.cpp", "src/train.cpp"])
t.wait()
if t.returncode == 1:
    sys.exit(1)

run_tests()

sys.exit(0)
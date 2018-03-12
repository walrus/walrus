#!/usr/bin/python

# noinspection PyUnresolvedReferences,PyUnresolvedReferences
import os, sys, subprocess

# Compile script for the new-network program, will recompile all dependencies
#
# This script should be run from linux/

#
# Main Program
#


# Check for being in linux/
_, cwd = os.path.split(os.getcwd())
if not cwd == "linux":
    print "Please run from the project/linux/ folder, not %s/" % cwd
    sys.exit(1)


# Parse arguments
# noinspection PyUnresolvedReferences
if len(sys.argv) > 1:
    print "Too many arguments given; try again."
    sys.exit(1)


# Compile the various source files
print "Compiling..."
a = subprocess.Popen(["g++", "-c", "-std=c++11", "../network/src/network-linux.cpp"])
b = subprocess.Popen(["g++", "-c", "-std=c++11", "../network/src/network-saveload-linux.cpp"])
c = subprocess.Popen(["g++", "-c", "-std=c++11", "src/new-network.cpp"])

a.wait()
if a.returncode == 1:
    sys.exit(1)
b.wait()
if b.returncode == 1:
    sys.exit(1)
c.wait()
if c.returncode == 1:
    sys.exit(1)


# Link the object files together into an executable
print "Linking..."
o = subprocess.Popen(["g++", "new-network.o", "../network/network-linux.o", "../network/network-saveload-linux.o", "-o", "new-network", "-std=c++11"])
o.wait()
if o.returncode == 1:
    sys.exit(1)

print "Success"
sys.exit(0)
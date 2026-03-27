#!/bin/bash

# Compile the nbody.cc file
g++ -O2 -o nbody nbody.cc

# Run nbody with different N values
for N in 128 256 512 1024 2048 4096; do
    echo "Running nbody with N=$N"
    ./nbody "$N"
done


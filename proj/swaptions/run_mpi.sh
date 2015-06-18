#!/bin/bash
thorq --add --mode mpi --slots 16 --nodes 4 --device gpu ./swaptions -ns 128 -sm 1000


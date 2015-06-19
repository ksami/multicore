#!/bin/bash
thorq --add --mode mpi --nodes 4 --device gpu ./swaptions -ns 128 -sm 10000


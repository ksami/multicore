#!/bin/bash
thorq --add --mode snucl --nodes 5 --device gpu ./swaptions -ns 128 -sm 100000

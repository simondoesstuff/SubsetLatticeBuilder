#!/bin/bash

if [ $# -le 0 ]; then
    echo "Usage: build_ancestral_net.sh <input> [output]"
    exit 1
fi

input=$1
output=$2

if [ ! -f $input ]; then
    echo "Input file $input does not exist"
    exit 1
fi

if [ -z $output ]; then
    output=${input%.*}.soln
fi

echo "Building ancestral network for $input > $output"
echo "Press enter to continue"
read

echo "Building rust program"
RUSTFLAGS="-C target-cpu=native" cargo build --release

echo "Running rust program"
target/release/trickle-constr $input $output
#!/bin/bash

for i in `seq 7 16`;
do
    if [ ! -d ../data/exp$i ]; then
        cd ../data/
        mkdir exp$i
        cd ../src/
    fi
    echo Running exp$i
    ./run.sh ../data/exp$i
done

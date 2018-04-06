#!/bin/bash

filepath=$1
savedir=$2

declare -a types=("company"
                "facility"
                "loc"
                "movie"
                "musicartist"
                "other"
                "person"
                "product"
                "sportsteam"
                "tvshow")
declare -a ref=("true"
                "predict")

for i in "${ref[@]}"
do
    for j in "${types[@]}"
    do
        echo ${i}, ${j}
        python -u predict_entity_analyzer.py --file ${filepath} --ref ${i} --entity ${j} > ${savedir}/${j}_ref-${i}
        python -u print_errors.py --read_file ${savedir}/${j}_ref-${i} --write_file ${savedir}/0_ref-${i}_${j}.xlsx
    done
done


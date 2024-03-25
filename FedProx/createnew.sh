#!/bin/bash

if [ -d "data" ]; then
    rm -r "data"
fi

user_names=($(jq -r '.User[]' configure.json))
classes=($(jq -r '.DataCategory[]' configure.json))

for user in "${user_names[@]}"; do
    for class in "${classes[@]}"; do
        mkdir -p "data/${user}/${class}"
    done
done

# shared
for class in "${classes[@]}"; do
	mkdir -p "data/shared/${class}"
done

python ./1-classification.py

DatasetName=($(jq -r '.DatasetName' configure.json))

if [ "$DatasetName" = "COVID" ]; then
    cp -r ../COVID_origindata/test ./data
elif [ "$DatasetName" = "CIFAR" ]; then
    cp -r ../CIFAR_origindata/test ./data
else
    echo "DatasetName is neither 'COVID' nor 'CIFAR'"
fi


python ./2-getinfo.py "$user_names_string" "$classes_string"

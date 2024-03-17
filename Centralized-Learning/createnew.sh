#!/bin/bash

if [ -d "data" ]; then
    rm -r "data"
fi

user_names="centralizedlearningdata"
classes=($(jq -r '.DataCategory[]' configure.json))

for user in "${user_names[@]}"; do
    for class in "${classes[@]}"; do
        mkdir -p "data/${user}/${class}"
    done
done

DatasetName=($(jq -r '.DatasetName' configure.json))

if [ "$DatasetName" = "COVID" ]; then
    cp -r ../COVID_origindata/COVID 			./data/centralizedlearningdata
	cp -r ../COVID_origindata/Lung_Opacity 		./data/centralizedlearningdata
    cp -r ../COVID_origindata/Normal 			./data/centralizedlearningdata
    cp -r ../COVID_origindata/VirtualPneumonia 	./data/centralizedlearningdata

    cp -r ../COVID_origindata/test ./data
elif [ "$DatasetName" = "CIFAR" ]; then
    cp -r ../CIFAR_origindata/0_airplane 	./data/centralizedlearningdata
	cp -r ../CIFAR_origindata/1_automobile 	./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/2_bird 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/3_cat 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/4_deer 		./data/centralizedlearningdata
	cp -r ../CIFAR_origindata/5_dog 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/6_frog 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/7_horse 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/8_ship 		./data/centralizedlearningdata
	cp -r ../CIFAR_origindata/9_truck 		./data/centralizedlearningdata
    cp -r ../CIFAR_origindata/test 			./data
else
    echo "DatasetName is neither 'COVID' nor 'CIFAR'"
fi


python ./2-getinfo.py "$user_names" "$classes_string"

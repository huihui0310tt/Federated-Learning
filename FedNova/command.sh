#!/bin/bash


# e.g. FedAvg-CIFAR-IID-ModelName
TaskName="FedAvg-CI"
rounds=100
epochs=5
# CIFAR-10=0.1, COVID=0.001
lr=0.1

# model= "ResNet18", "MobileNet_v2", "shufflenet_v2_x2_0"
model="shufflenet_v2_x2_0"

#Data= "CIFAR" or "COVID"
dataset="CIFAR"

batch_size=64

# cuda:0 or cuda:1
CUDA_Name="cuda:0"




python main.py --TaskName="$TaskName"   --rounds="$rounds"   --epochs="$epochs"  --model="$model"  --dataset="$dataset"   --lr="$lr"  --batch_size="$batch_size"   --CUDA_Name="$CUDA_Name"

rm -r ./data

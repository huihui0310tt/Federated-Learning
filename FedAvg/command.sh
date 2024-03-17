#!/bin/bash


# e.g. FedAvg-CIFAR-IID-ModelName
TaskName="FedAvg-CI"
rounds=100
epochs=5
# CIFAR-10=0.1, COVID=0.001
lr=0.1

# model : "ResNet18", "VGG16", "MobileNet_v2"
model="ResNet18"

#Data : "CIFAR" or "COVID"
dataset="COVID"

batch_size=64

# cuda:0 or cuda:1
CUDA_Name="cuda:0"




python main.py --TaskName="$TaskName"   --rounds="$rounds"   --epochs="$epochs"  --model="$model"  --dataset="dataset"   --lr="$lr"  --batch_size="$batch_size"   --CUDA_Name="$CUDA_Name"

rm -r ./data
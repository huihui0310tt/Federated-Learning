#!/bin/bash


# e.g. CIFAR-CL-ResNet18
TaskName="CIFAR-CL"

epochs=2
# CIFAR-10=0.1, COVID=0.001
lr=0.1

# model= "ResNet18", "MobileNet_v2", "shufflenet_v2_x2_0"
model="ResNet18"

#Data= "CIFAR" or "COVID"
dataset="CIFAR"

# Default = 64
batch_size=128


# cuda:0 or cuda:1
CUDA_Name="cuda:0"




python main.py --TaskName="$TaskName"      --epochs="$epochs"  --model="$model"  --dataset="$dataset"   --lr="$lr"  --batch_size="$batch_size"   --CUDA_Name="$CUDA_Name"

rm -r ./data

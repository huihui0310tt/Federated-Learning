#!/bin/bash


# e.g. CIFAR-FedAvg-IID
TaskName="CIFAR10-"
rounds=100
epochs=5
# CIFAR-10=0.1, COVID=0.001
lr=0.1

batch_size=64

# cuda:0 or cuda:1
CUDA_Name="cuda:0"




python main.py --TaskName="$TaskName"   --rounds="$rounds"   --epochs="$epochs"   --lr="$lr"   --batch_size="$batch_size"   --CUDA_Name="$CUDA_Name"

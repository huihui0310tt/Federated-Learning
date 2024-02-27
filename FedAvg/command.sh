#!/bin/bash



TaskName="CIFAR10-"
rounds=2
epochs=2
lr=0.1
batch_size=256
CUDA_Name="cuda:0"




python main.py --TaskName="$TaskName"   --rounds="$rounds"   --epochs="$epochs"   --lr="$lr"   --batch_size="$batch_size"   --CUDA_Name="$CUDA_Name"

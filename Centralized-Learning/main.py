import argparser
import client_base
import csv
import time
from datetime import datetime
import json
import torch

def federated_learning():
    ###################################################################            # Initial
    TaskName, epochs, model, dataset, lr, batch_size, global_model, num_classes, no_cuda, gpu_devicename = argparser.get_arg()
    client = client_base.Client('centralizedlearningdata')

    
    ###################################################################
    print('TaskName: ', TaskName)
    print('epochs : ', epochs)
    print('model : ', model )
    print('dataset : ', dataset )
    print('lr : ', lr)
    print('batch_size : ', batch_size)
    print('global_model : ', global_model)
    print('no_cuda : ', no_cuda)
    print('gpu_devicename : ', gpu_devicename)

    print(str(datetime.now()))

    record = []
    acc_record = []
    loss_record = []

    client.train(epochs, global_model, lr, model, dataset, batch_size, num_classes, no_cuda, gpu_devicename)
    record = client.metrics

    for i in record:
        acc_record.append([i['accuracy']])
        loss_record.append([i['loss']])

    print(acc_record)
    print(loss_record)
        
    writefile(acc_record, loss_record, ['cl_user'], TaskName)

    print('Train Finish')


def writefile(acc_record, loss_record, fieldnames, TaskName):
    nowtime = str(datetime.now())
    acc_filename = './result_save/' + TaskName + ' ' + nowtime + ' accuracy.csv'
    loss_filename = './result_save/' + TaskName + ' ' + nowtime + ' loss.csv'
    
    with open(acc_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for record in acc_record:
            row_data = {field: value for field, value in zip(fieldnames, record)}
            writer.writerow(row_data)

    with open(loss_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for record in loss_record:
            row_data = {field: value for field, value in zip(fieldnames, record)}
            writer.writerow(row_data)
if __name__ == "__main__":
    # federated_learning(epoch=1)
    federated_learning()

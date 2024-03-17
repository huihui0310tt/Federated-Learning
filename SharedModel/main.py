import threading
import argparser
import client_base
import aggregator_base
import csv
import time
from datetime import datetime
import json
import torch

def federated_learning():
    ###################################################################            # Initial
    TaskName, user_list, rounds, epochs, model, dataset, lr, batch_size, global_model, num_classes, no_cuda, gpu_devicename = argparser.get_arg()
    clients = []
    for user_name in user_list:
        clients.append(client_base.Client(user_name))
    shared_model = client_base.Client('shared')
    aggregator = aggregator_base.Aggregator()
    
    ###################################################################
    print('TaskName: ', TaskName)
    print('user : ', user_list)
    print('round : ', rounds)
    print('epochs : ', epochs)
    print('model : ', model )
    print('dataset : ', dataset )
    print('lr : ', lr)
    print('batch_size : ', batch_size)
    print('global_model : ', global_model)
    print('no_cuda : ', no_cuda)
    print('gpu_devicename : ', gpu_devicename)

    acc_record = []
    loss_record = []
    best_accuracy = 0
    
    for round_idx in range(rounds):
        print('----------------------------Now Round ', str(round_idx + 1), '----------------------------')
        print(str(datetime.now()))
 

        one_round_acc_record = []
        one_round_loss_record = []
        best_accuracy = 0 

        shared_model.train(epochs, global_model, lr, model, dataset, batch_size, num_classes, no_cuda, gpu_devicename, shared_data_train=True)
        global_model = shared_model.model
        one_round_acc_record.append(shared_model.metrics[-1]['accuracy'])
        one_round_loss_record.append(shared_model.metrics[-1]['loss'])
        
        
        for i in clients:
            if global_model is None:
                print('global: ', global_model)
            i.train(epochs, global_model, lr, model, dataset, batch_size, num_classes, no_cuda, gpu_devicename)
            one_round_acc_record.append(i.metrics[-1]['accuracy'])
            one_round_loss_record.append(i.metrics[-1]['loss'])


        print('---Global model---')
        clients.append(shared_model)           

        metrics, global_model = aggregator.merge(clients, model, dataset, batch_size, num_classes, no_cuda, gpu_devicename)
        clients.pop(-1)

        one_round_acc_record.append(metrics['accuracy'])
        one_round_loss_record.append(metrics['loss'])
        
        acc_record.append(one_round_acc_record)
        loss_record.append(one_round_loss_record)
        
        if metrics['accuracy'] > best_accuracy :
            best_accuracy = metrics['accuracy']
            torch.save(global_model, './result_save/aggregate_best')



    writefile(acc_record, loss_record, user_list, TaskName)

    print('Train Finish')


def writefile(acc_record, loss_record, fieldnames, TaskName):
    fieldnames.insert(0, 'sharedmodel')
    fieldnames.append('aggregator')
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

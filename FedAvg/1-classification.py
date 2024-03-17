import shutil
from os import listdir
from os.path import isfile, isdir, join
import json


file_path = 'configure.json'
json_data = None
sample = []
with open(file_path, 'r') as file:
    json_data = json.load(file)
des = None
label = None




des = json_data['User']
label = json_data['DataCategory']
datadistribution = json_data['DataDistribution']
shared_data        = datadistribution[0]['shared']
# client1_samplesize = [726,  	1195,	2019,	269]
# client2_samplesize = [717, 	1202,  	2045,  	268]
# client3_samplesize = [714,  	1217,   2063,   268]
# client4_samplesize = [735, 	1195,  	2026,  	271]
DatasetName = json_data['DatasetName']
# DatasetName = 'CIFAR' or 'COVID'
if DatasetName == 'CIFAR': 
    src = '../CIFAR_origindata'
elif DatasetName == 'COVID':
    src = '../COVID_origindata'
#src = "./origindata"
shared = "./data/shared"
for index in range(len(des)):
    sample.append(datadistribution[index+1][des[index]])

for label_idx in range(len(label)):
    pass
    count = 1
    for one_src_file in listdir(join(src, label[label_idx])) :
        if count <= shared_data[label_idx] :
            #print(str(join(src, label[label_idx], one_src_file)), str(join(shared, label[label_idx], one_src_file)))
            shutil.copy(join(src, label[label_idx], one_src_file), join(shared, label[label_idx], one_src_file))

        else:
            temp = shared_data[label_idx]
            for client_idx in range(len(sample)) :
                temp += sample[client_idx][label_idx]
                if count <= temp:
                    #print(str(join(src, label[label_idx], one_src_file)), str(join("./data",des[client_idx], label[label_idx])))
                    shutil.copy(join(src, label[label_idx], one_src_file), join("./data", des[client_idx], label[label_idx]))
                    break
        count+=1




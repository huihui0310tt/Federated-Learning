from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


import copy
# from net import resnet18
from torchvision.models import vgg16, resnet18, mobilenet_v2, alexnet


class Aggregator:

    def __test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += nn.CrossEntropyLoss()(output, target)

                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        metrics = {
            'loss': test_loss,
            'accuracy': (100. * correct / len(test_loader.dataset))
        }
        return metrics


    def merge(self, clients, modelname, dataset, batch_size, num_classes, no_cuda, gpu_devicename):
        # weights = [torch.load(m['path'], 'cpu') for m in models]
        weights = [client.model for client in clients]
        
        # total_data_size = sum(m['size'] for m in models)
        total_data_size = sum(client.sample for client in clients)

        # factors = [m['size'] / total_data_size for m in models]
        factors = [client.sample/total_data_size for client in clients]

        merged = {}
        for key in weights[0].keys():
            merged[key] = sum([w[key] * f for w, f in zip(weights, factors)])


        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device( gpu_devicename if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        if dataset == 'CIFAR':
            test_data = datasets.ImageFolder('./data/test',
                                            transforms.Compose([transforms.Resize((128, 128)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                    std=[0.247, 0.243, 0.261])
                                                                ]))
        elif dataset == 'COVID':
            test_data = datasets.ImageFolder('./data/test',
                                            transforms.Compose([transforms.Resize((224, 224)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])
                                                                ]))
        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                **kwargs)
        model = None
        # modelname : "ResNet18", "VGG16", "MobileNet_v2", "AlexNet"

        if modelname == 'ResNet18':
            model = resnet18(num_classes=num_classes, pretrained=False).to(device)
        elif modelname == 'VGG16':
            model = vgg16(num_classes=num_classes, pretrained=False).to(device)
        elif modelname == 'MobileNet_v2':
            model = mobilenet_v2(num_classes=num_classes, pretrained=False).to(device)
        elif modelname =='AlexNet':
            model = alexnet(num_classes=num_classes, pretrained=False).to(device)
        # model = resnet18(num_classes=num_classes).to(device)
        model.load_state_dict(copy.deepcopy(merged))
        metrics = self.__test(model, device, test_loader)
        print(metrics)
        torch.save(model, './result_save/aggregate_final')

        return metrics, merged
        

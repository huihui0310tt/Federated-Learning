import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from net import resnet18


class Client:
    def __init__(self, name):
        self.name = name
        self.sample = None
        self.model = None
        self.metrics = None
        self.resume = './data/' + str(name)


    def __train(self, model, device, train_loader, optimizer):
        model.train()
        loss_total = 0
        counter = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # loss = nn.NLLLoss()(output, target)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            counter +=1
        
        return loss_total/counter

    def __test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # test_loss += nn.NLLLoss()(output, target)
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



    def train(self, epochs, global_model, lr, batch_size, num_classes, no_cuda, gpu_devicename):

        seed = 1

        use_cuda = not no_cuda and torch.cuda.is_available()
        torch.manual_seed(seed)
        device = torch.device(gpu_devicename if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        training_data = datasets.ImageFolder(self.resume,
                                            transforms.Compose([transforms.Resize((128, 128)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                    std=[0.247, 0.243, 0.261]),
                                                                transforms.RandomErasing(scale=(0.02, 0.1))
                                                                ]))

        test_data = datasets.ImageFolder('./data/test',
                                        transforms.Compose([transforms.Resize((128, 128)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                                std=[0.247, 0.243, 0.261])
                                                            ]))

        test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                **kwargs)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(
            training_data, range(len(training_data))),
            batch_size=batch_size,
            shuffle=True,
            **kwargs)

        model = resnet18(num_classes=num_classes).to(device)
        # model = vgg16().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0 )
        
        if global_model is not None:
            model.load_state_dict(global_model)

        model.train()

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
        log_metrics = []
        metrics = {}
        for epoch in range(1, epochs + 1):
            training_loss = self.__train(model, device, train_loader, optimizer)
            metrics = self.__test(model, device, test_loader)
            metrics['loss'] = training_loss
            scheduler.step()
            log_metrics.append(metrics)
              

        self.sample = len(training_data)
        self.model = model.state_dict()
        self.metrics = log_metrics

        print(self.name, '\t', end='')
        print(log_metrics[-1])
        print()

        torch.save(model, './result_save/' + self.name)

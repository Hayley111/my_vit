from __future__ import print_function
import time
import os
from random import shuffle
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.build_model import build_model

def train(model, device, train_loader, optimizer, epoch):
    m = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()
    model.train()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  

        output = model(data)
        loss = loss_fn(m(output),target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        grad_accumulate = 1
        if batch_idx % grad_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
        if batch_idx % 2 == 0 and device ==0 :
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    torch.cuda.synchronize()
    start_time = time.time()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()


    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('./data', train=True, 
                                    download=False,
                                    transform=transform)
                                    # target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    dataset_val = datasets.MNIST('./data', train=False, 
                                    download=False,
                                    transform=transform)
                                    # target_transform=transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, 
                                                        num_replicas=num_tasks, 
                                                        rank=global_rank,
                                                        shuffle = True)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                sampler=sampler_train,
                                                batch_size = 64,
                                                num_workers= 8,
                                                pin_memory = True)
    test_loader = torch.utils.data.DataLoader(dataset_val,
                                                sampler=sampler_val,
                                                batch_size = 16,
                                                num_workers= 8,
                                                pin_memory = True)
                                                
    print('train_loader len:',len(train_loader),"test_loader len:", len(test_loader))

    device = rank % torch.cuda.device_count()
    # TODO: test jit script to improve performance
    # model = torch.jit.script(Net().to(device))

    model = build_model('vit').to(device)
    # save model description
    summary(build_model('vit'), (1, 224, 224), device='cpu')



    ddp_model = DDP(model, device_ids=[device])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(0, 16):
        train(ddp_model, device, train_loader, optimizer, epoch)
        test(ddp_model, device, test_loader)
        scheduler.step()
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    print(f'****total time****:{total_time}')

    if device == 0:
        savefile = str(f"mnist_cnn_{device}.pt")
        torch.save(model.state_dict(), savefile)


if __name__ == '__main__':
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1234'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    torch.manual_seed(dist.get_rank())
    print(f"Start running basic DDP example on rank {rank}.")
    main()
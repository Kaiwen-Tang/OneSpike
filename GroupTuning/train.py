from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from catSNN import SpikeDataset , fuse_bn_recursively

from utils_vgg16 import transfer_model_vgg16,test, data_loader, train
import logging


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 LeNet Example')
    parser.add_argument('--testBatch', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', type=str, default=None, metavar='SAVE',
                        help='For Saving the current Model')
    parser.add_argument('--resume', type=str, default=None, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--logdir', type=str, default=None,
                        help='the dir of log')
    parser.add_argument('--timestep', type=int, nargs='+', default=[16,16,16], metavar='N',
                    help='SNN time window (list of ints)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # ----------log--------------
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(args.logdir)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('my_logger')
    logger.addHandler(file_handler)
    logger.info('Log for testing model code')
    # ----------------------------

    train_loader, val_loader , val_dataset= data_loader(timestep = args.timestep[0])
    snn_dataset = SpikeDataset(val_dataset, T = args.timestep[0],theta = 1-0.0001)
    snn_loader = torch.utils.data.DataLoader(snn_dataset, batch_size=args.testBatch, shuffle=False)

    from VGG_flex import VGG_flex
    from VGG_onespike_flex import VGGos_flex
    model1 = VGG_flex('VGG16',args.timestep, bias = True).to(device)
    model1.load_state_dict(torch.load(args.resume), strict=False)
    correct_ = 0
    optimizer = optim.SGD(model1.parameters(), lr=args.lr,momentum = 0.9,weight_decay= 1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    k = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model1, device, train_loader, optimizer, epoch)
        correct = test(model1, device, val_loader)
        if correct>correct_:
            correct_ = correct
            torch.save(model1.state_dict(), args.save)
        k+=1
        scheduler.step()

    correct = test(model1, device, val_loader)

    model1 = fuse_bn_recursively(model1)
    snn_model_PSG = VGGos_flex('VGG16', args.timestep, bias =True).to(device)
    snn_model_PSG = transfer_model_vgg16(model1, snn_model_PSG)
    print("Test snn model PSG")
    test(snn_model_PSG, device, snn_loader)
    
    
if __name__ == '__main__':
    main()

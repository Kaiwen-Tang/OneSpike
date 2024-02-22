import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

class AddQuantization(object):
    def __init__(self, timestep, min=0., max=1.):
        self.min = min
        self.max = max
        self.timestep = timestep
        
    def __call__(self, tensor):
        return torch.clamp(torch.div(torch.floor(torch.mul(tensor, self.timestep)), self.timestep),min=0, max=1)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def data_loader(batch_size=300, timestep=64, workers=1, pin_memory=True):
    traindir = os.path.join('../../../../ImageNet/train')
    valdir = os.path.join('../../../../ImageNet/val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(10),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
            AddQuantization(timestep=timestep)
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            AddQuantization(timestep=timestep)
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, val_dataset

def transfer_model_vgg16(model1, snn_model_PSG):
    src_dict = model1.state_dict()
    dst_dict_PSG = snn_model_PSG.state_dict()
    f = open('dstname.txt','w')
    for i in dst_dict_PSG.keys():
        f.write(str(i)+'\n')
    f.close()
    f = open('srcname.txt','w')
    for i in src_dict.keys():
        f.write(str(i)+'\n')
    f.close()
    reshape_dict = {}
    reshape_dict['features.0.conv.weight'] = torch.nn.Parameter(src_dict['features.0.weight'].reshape(dst_dict_PSG['features.0.conv.weight'].shape))
    reshape_dict['features.0.conv.bias'] = torch.nn.Parameter(src_dict['features.0.bias'].reshape(dst_dict_PSG['features.0.conv.bias'].shape))
    reshape_dict['features.1.conv.weight'] = torch.nn.Parameter(src_dict['features.2.weight'].reshape(dst_dict_PSG['features.1.conv.weight'].shape))
    reshape_dict['features.1.conv.bias'] = torch.nn.Parameter(src_dict['features.2.bias'].reshape(dst_dict_PSG['features.1.conv.bias'].shape))
    reshape_dict['features.2.pool.weight'] = dst_dict_PSG['features.2.pool.weight']
    reshape_dict['features.3.conv.weight'] = torch.nn.Parameter(src_dict['features.6.weight'].reshape(dst_dict_PSG['features.3.conv.weight'].shape))
    reshape_dict['features.3.conv.bias'] = torch.nn.Parameter(src_dict['features.6.bias'].reshape(dst_dict_PSG['features.3.conv.bias'].shape))
    reshape_dict['features.4.conv.weight'] = torch.nn.Parameter(src_dict['features.8.weight'].reshape(dst_dict_PSG['features.4.conv.weight'].shape))
    reshape_dict['features.4.conv.bias'] = torch.nn.Parameter(src_dict['features.8.bias'].reshape(dst_dict_PSG['features.4.conv.bias'].shape))
    reshape_dict['features.5.pool.weight'] = dst_dict_PSG['features.5.pool.weight']
    reshape_dict['features.6.conv.weight'] = torch.nn.Parameter(src_dict['features.12.weight'].reshape(dst_dict_PSG['features.6.conv.weight'].shape))
    reshape_dict['features.6.conv.bias'] = torch.nn.Parameter(src_dict['features.12.bias'].reshape(dst_dict_PSG['features.6.conv.bias'].shape))
    reshape_dict['features.7.conv.weight'] = torch.nn.Parameter(src_dict['features.14.weight'].reshape(dst_dict_PSG['features.7.conv.weight'].shape))
    reshape_dict['features.7.conv.bias'] = torch.nn.Parameter(src_dict['features.14.bias'].reshape(dst_dict_PSG['features.7.conv.bias'].shape))
    reshape_dict['features.8.conv.weight'] = torch.nn.Parameter(src_dict['features.16.weight'].reshape(dst_dict_PSG['features.8.conv.weight'].shape))
    reshape_dict['features.8.conv.bias'] = torch.nn.Parameter(src_dict['features.16.bias'].reshape(dst_dict_PSG['features.8.conv.bias'].shape))
    reshape_dict['features.9.pool.weight'] = dst_dict_PSG['features.9.pool.weight']
    reshape_dict['features.10.conv.weight'] = torch.nn.Parameter(src_dict['features.20.weight'].reshape(dst_dict_PSG['features.10.conv.weight'].shape))
    reshape_dict['features.10.conv.bias'] = torch.nn.Parameter(src_dict['features.20.bias'].reshape(dst_dict_PSG['features.10.conv.bias'].shape))
    reshape_dict['features.11.conv.weight'] = torch.nn.Parameter(src_dict['features.22.weight'].reshape(dst_dict_PSG['features.11.conv.weight'].shape))
    reshape_dict['features.11.conv.bias'] = torch.nn.Parameter(src_dict['features.22.bias'].reshape(dst_dict_PSG['features.11.conv.bias'].shape))
    reshape_dict['features.12.conv.weight'] = torch.nn.Parameter(src_dict['features.24.weight'].reshape(dst_dict_PSG['features.12.conv.weight'].shape))
    reshape_dict['features.12.conv.bias'] = torch.nn.Parameter(src_dict['features.24.bias'].reshape(dst_dict_PSG['features.12.conv.bias'].shape))
    reshape_dict['features.13.pool.weight'] = dst_dict_PSG['features.13.pool.weight']
    reshape_dict['features.14.conv.weight'] = torch.nn.Parameter(src_dict['features.28.weight'].reshape(dst_dict_PSG['features.14.conv.weight'].shape))
    reshape_dict['features.14.conv.bias'] = torch.nn.Parameter(src_dict['features.28.bias'].reshape(dst_dict_PSG['features.14.conv.bias'].shape))
    reshape_dict['features.15.conv.weight'] = torch.nn.Parameter(src_dict['features.30.weight'].reshape(dst_dict_PSG['features.15.conv.weight'].shape))
    reshape_dict['features.15.conv.bias'] = torch.nn.Parameter(src_dict['features.30.bias'].reshape(dst_dict_PSG['features.15.conv.bias'].shape))
    reshape_dict['features.16.conv.weight'] = torch.nn.Parameter(src_dict['features.32.weight'].reshape(dst_dict_PSG['features.16.conv.weight'].shape))
    reshape_dict['features.16.conv.bias'] = torch.nn.Parameter(src_dict['features.32.bias'].reshape(dst_dict_PSG['features.16.conv.bias'].shape))
    reshape_dict['classifier0.weight'] = torch.nn.Parameter(src_dict['classifier0.weight'].reshape(dst_dict_PSG['classifier0.weight'].shape))
    reshape_dict['classifier0.bias'] = torch.nn.Parameter(src_dict['classifier0.bias'].reshape(dst_dict_PSG['classifier0.bias'].shape))
    reshape_dict['classifier3.weight'] = torch.nn.Parameter(src_dict['classifier3.weight'].reshape(dst_dict_PSG['classifier3.weight'].shape))
    reshape_dict['classifier3.bias'] = torch.nn.Parameter(src_dict['classifier3.bias'].reshape(dst_dict_PSG['classifier3.bias'].shape))
    reshape_dict['classifier6.weight'] = torch.nn.Parameter(src_dict['classifier6.weight'].reshape(dst_dict_PSG['classifier6.weight'].shape))
    reshape_dict['classifier6.bias'] = torch.nn.Parameter(src_dict['classifier6.bias'].reshape(dst_dict_PSG['classifier6.bias'].shape))
    snn_model_PSG.load_state_dict(reshape_dict, strict=False)
    return snn_model_PSG

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # print(pred.eq(target.view_as(pred)).sum().item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


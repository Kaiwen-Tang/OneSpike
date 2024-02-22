
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import catSNN
import catCuda

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M']
}
    
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timestep):
        timestep_f = float(timestep)
        return torch.div(torch.floor(torch.mul(input, timestep_f)), timestep_f)

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output), None

quan_my = STEFunction.apply

# Clamp_q_(timestep)
class Clamp_q_(nn.Module):
    def __init__(self, min=0.0, max=1.0, timestep=2):
        super(Clamp_q_, self).__init__()
        self.min = min
        self.max = max
        self.timestep_f = float(timestep)

    def forward(self, x):
        x = torch.clamp(x, min=self.min, max=self.max)
        x = quan_my(x, self.timestep_f)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.normal_(m.weight, 0, 0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
class VGG_flex(nn.Module):
    def __init__(self, vgg_name, timestep, quantize_factor=-1, clamp_max = 1.0, bias=False, quantize_bit=32):
        super(VGG_flex, self).__init__()
        self.timestep = timestep
        self.quantize_factor=quantize_factor
        self.clamp_max = clamp_max
        self.bias = bias
        self.relu0 = Clamp_q_(timestep = self.timestep[19])
        self.relu1 = Clamp_q_(timestep = self.timestep[20])
        self.features = self._make_layers(cfg[vgg_name], quantize_bit=quantize_bit)
        self.classifier0 = nn.Linear(512 * 7 * 7, 4096, bias=True)
        self.classifier3 = nn.Linear(4096, 4096, bias=True)
        self.classifier6 = nn.Linear(4096, 1000, bias=True)
        self.features.apply(initialize_weights)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier0(out)
        out = self.relu0(out)
        out = self.classifier3(out)
        out = self.relu1(out)
        out = self.classifier6(out)
        return out
 

    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    def _make_layers(self, cfg, quantize_bit=32):
        layers = []
        in_channels = 3
        index=0
        for x in cfg:
            index+=1 #From 1 to 18
            if x == 'M':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2),Clamp_q_(timestep = self.timestep[index-1])]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),nn.BatchNorm2d(x),
                           Clamp_q_(timestep = self.timestep[index-1])]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1),Clamp_q_(timestep = self.timestep[index])]
        return nn.Sequential(*layers)

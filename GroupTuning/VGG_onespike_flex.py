
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn.functional as F
import torch.nn as nn
import catSNN
import catCuda

T_reduce = 2



cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_': [64, 64, (64,64), 128, 128, (128,128), 256, 256, 256, 256, (256,256), 512, 512, 512, 512, (512,512), 512, 512, 512, 512, (512,512)],
    'Mynetwork':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,512,512, 'M']
}

def get_spike(x_out,threshold = 1, p=4):
    spike_out = []
    for i in range(p):
        input_i = (i* x_out) % (threshold-0.001) + x_out
        new_spike = catCuda.getSpikes(input_i.type(torch.FloatTensor).cuda(), threshold - 0.001)
        spike_out.append(new_spike)
    x=torch.cat(spike_out, dim=-1)
    return x
    
class SpikeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True, T_layer=T_reduce, T_nxt=T_reduce):
        super(SpikeLayer, self).__init__()
        self.T_layer = T_layer
        self.T_nxt = T_nxt
        snn = catSNN.spikeLayer(T_layer)
        self.conv = snn.conv(inChannels=in_channels, outChannels=out_channels, kernelSize=kernel_size,
                                      padding=padding, bias=bias)
    def forward(self, x):
        x_tup = torch.chunk(x,self.T_layer,dim=-1)
        x_out = []
        for i in range(self.T_layer):
            x_out.append(self.conv(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0)
        x = get_spike(x_out,1,self.T_nxt)
        return x

class SpikePool(nn.Module):
    def __init__(self, T_layer=T_reduce, T_nxt=T_reduce):
        super(SpikePool, self).__init__()
        self.T_layer = T_layer
        self.T_nxt = T_nxt
        snn = catSNN.spikeLayer(T_layer)
        self.snn=snn
        self.pool = self.snn.pool(2)

    def forward(self, x):
        x_tup = torch.chunk(x,self.T_layer,dim=-1)
        x_out = []
        for i in range(self.T_layer):
            x_out.append(self.pool(x_tup[i]))
        
        x_out = torch.mean(torch.stack(x_out),dim=0)
        x_out = get_spike(x_out,1,self.T_nxt)
        return x_out
    

class VGGos_flex(nn.Module):
    def __init__(self, vgg_name, timestep, bias=False):
        super(VGGos_flex, self).__init__()
        self.timestep = timestep
        self.snn0 = catSNN.spikeLayer(timestep[18])
        self.snn1 = catSNN.spikeLayer(timestep[19])
        self.snn2 = catSNN.spikeLayer(timestep[20])
        # self.T = T_reduce
        self.bias=bias
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier0 = self.snn0.dense((7,7,512),4096, bias=True)
        self.classifier3 = self.snn1.dense((1,1,4096), 4096, bias=True)
        self.classifier6 = self.snn2.dense((1,1,4096), 1000, bias=True)

    def forward(self, x):
        out = self.features(x)

        x_tup = torch.chunk(out,self.timestep[18],dim=-1)
        x_out = []
        for i in range(self.timestep[18]):
            x_out.append(self.classifier0(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0)
        out = get_spike(x_out,1,self.timestep[18])

        x_tup = torch.chunk(out,self.timestep[19],dim=-1)
        x_out = []
        for i in range(self.timestep[19]):
            x_out.append(self.classifier3(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0)
        out = get_spike(x_out,1,self.timestep[19])

        x_tup = torch.chunk(out,self.timestep[20],dim=-1)
        x_out = []
        for i in range(self.timestep[20]):
            x_out.append(self.classifier6(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0).squeeze()
 
        # out = self.classifier6(out)
        # out = self.snn.sum_spikes(out)/self.timestep[20]
        return x_out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index=0
        for x in cfg:
            index+=1 #From 1 to 18
            # [64, 64, 'M（3）', 128, 128, 'M（6）', 256, 256, 256, 'M（10）', 512, 512, 512, 'M', 512, 512, 512, 'M']
            if x == 'M':
                layers += [SpikePool(T_layer=self.timestep[index-1], T_nxt=self.timestep[index])]
            else:
                layers += [SpikeLayer(in_channels, x, bias=self.bias, T_layer=self.timestep[index-1], T_nxt=self.timestep[index])]
                in_channels = x
        return nn.Sequential(*layers)

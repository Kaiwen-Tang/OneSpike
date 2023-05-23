import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from se_block import SEBlock
import torch
import numpy as np
import catCuda
import catSNN
Clp_max = 3
T = 8
T_reduce = 24

def get_spike(x_out,threshold = 1, p=4):
    spike_out = []
    current_potential, tmp_potential = x_out, x_out
    for i in range(p):
        spike_out.append(catCuda.getSpikes(current_potential.type(torch.FloatTensor).cuda(), threshold - 0.001))
        if i == p-1:
            break
        tmp_potential = torch.where(current_potential > threshold, current_potential - threshold, current_potential)
        current_potential = tmp_potential + x_out
    x=torch.cat(spike_out, dim=-1)
    return x

def create_spike_input_cuda(input,T):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    # 1/2.5:max_1-0.0001; 8/10 : max_1-0.001 ; 16/20 : max_1-0.0001 ; 32/40 : max_1-0.001 ;
    out = catCuda.getSpikes(out, Clp_max - 0.001)
    return out

class RepVGGplusBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 deploy=False,
                 use_post_se=False):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.T = T_reduce
        snn = catSNN.spikeLayer(T_reduce)
        self.snn=snn

        assert kernel_size == 3
        assert padding == 1

        # self.nonlinearity = nn.ReLU()

        if use_post_se:
            self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            # self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            #                          padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.rbr_reparam = snn.conv(inChannels=in_channels, outChannels=out_channels, kernelSize=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=out_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x):
        x_tup = torch.chunk(x,T_reduce,dim=-1)
        x_out = []
        for i in range(T_reduce):
            x_out.append(self.rbr_reparam(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0)
        x = get_spike(x_out,Clp_max,T_reduce)
        return self.post_se(x)


class RepVGGplusStage(nn.Module):

    def __init__(self, in_planes, planes, num_blocks, stride, use_checkpoint, use_post_se=False, deploy=False):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = nn.ModuleList(blocks)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class RepVGGplus(nn.Module):
    """RepVGGplus
        An official improved version of RepVGG (RepVGG: Making VGG-style ConvNets Great Again) <https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf>`_.

        Args:
            num_blocks (tuple[int]): Depths of each stage.
            num_classes (tuple[int]): Num of classes.
            width_multiplier (tuple[float]): The width of the four stages
                will be (64 * width_multiplier[0], 128 * width_multiplier[1], 256 * width_multiplier[2], 512 * width_multiplier[3]).
            deploy (bool, optional): If True, the model will have the inference-time structure.
                Default: False.
            use_post_se (bool, optional): If True, the model will have Squeeze-and-Excitation blocks following the conv-ReLU units.
                Default: False.
            use_checkpoint (bool, optional): If True, the model will use torch.utils.checkpoint to save the GPU memory during training with acceptable slowdown.
                Do not use it if you have sufficient GPU memory.
                Default: False.
        """
    def __init__(self,
                 num_blocks,
                 num_classes,
                 width_multiplier,
                 deploy=False,
                 use_post_se=False,
                 use_checkpoint=False):
        super().__init__()

        self.deploy = deploy
        self.num_classes = num_classes
        self.T = T_reduce
        snn = catSNN.spikeLayer(T_reduce)
        self.snn=snn

        in_channels = min(64, int(64 * width_multiplier[0]))
        stage_channels = [int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), int(512 * width_multiplier[3])]
        self.stage0 = RepVGGplusBlock(in_channels=3, out_channels=in_channels, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_post_se=use_post_se)
        self.stage1 = RepVGGplusStage(in_channels, stage_channels[0], num_blocks[0], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage2 = RepVGGplusStage(stage_channels[0], stage_channels[1], num_blocks[1], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        #   split stage3 so that we can insert an auxiliary classifier
        self.stage3_first = RepVGGplusStage(stage_channels[1], stage_channels[2], num_blocks[2] // 2, stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage3_second = RepVGGplusStage(stage_channels[2], stage_channels[2], num_blocks[2] - num_blocks[2] // 2, stride=1, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        self.stage4 = RepVGGplusStage(stage_channels[2], stage_channels[3], num_blocks[3], stride=2, use_checkpoint=use_checkpoint, use_post_se=use_post_se, deploy=deploy)
        # self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.gap = self.snn.pool(10)

        self.flatten = nn.Flatten()
        # self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)
        self.linear = snn.dense(int(512 * width_multiplier[3]), num_classes, bias=True)

    def _build_aux_for_stage(self, stage):
        stage_out_channels = list(stage.blocks.children())[-1].rbr_dense.conv.out_channels
        downsample = conv_bn_relu(in_channels=stage_out_channels, out_channels=stage_out_channels, kernel_size=3, stride=2, padding=1)
        fc = nn.Linear(stage_out_channels, self.num_classes, bias=True)
        return nn.Sequential(downsample, nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3_first(out)
        out = self.stage3_second(out)
        out = self.stage4(out)
        x_tup = torch.chunk(out,T_reduce,dim=-1)
        x_out = []
        for i in range(T_reduce):
            x_out.append(self.gap(x_tup[i]))
        x_out = torch.mean(torch.stack(x_out),dim=0)
        y = get_spike(x_out,Clp_max,T_reduce)
        y = self.linear(y)
        y = self.snn.sum_spikes(y)/T_reduce
        return {
            'main': y,
        }

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True


#   torch.utils.checkpoint can reduce the memory consumption during training with a minor slowdown. Don't use it if you have sufficient GPU memory.
#   Not sure whether it slows down inference
#   pse for "post SE", which means using SE block after ReLU
def create_RepVGGplus_L2pse(deploy=False, use_checkpoint=False):
    return RepVGGplus(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], deploy=deploy, use_post_se=False,
                      use_checkpoint=use_checkpoint)

#   Will release more
repvggplus_func_dict = {
    'RepVGGplus-L2pse': create_RepVGGplus_L2pse,
}

def create_RepVGGplus_by_name(name, deploy=False, use_checkpoint=False):
    if 'plus' in name:
        return repvggplus_func_dict[name](deploy=deploy, use_checkpoint=use_checkpoint)
    else:
        print('=================== Building the vanila RepVGG ===================')
        from repvgg import get_RepVGG_func_by_name
        return get_RepVGG_func_by_name(name)(deploy=deploy, use_checkpoint=use_checkpoint)



def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    import copy
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

# class CBAM(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)),
#             nn.Conv3d(input_dim, output_dim, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.AdaptiveAvgPool3d((16, 16, 1)),
#             nn.Conv3d(input_dim, output_dim, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Sigmoid()
#         )
       
#     def forward(self, x):
#         # Channel attention
#         channel_attention = self.channel_attention(x)
#         channel_attention = channel_attention.expand_as(x)
#         x = x * channel_attention

#         # Spatial attention
#         spatial_attention = self.spatial_attention(x)
#         spatial_attention = spatial_attention.expand_as(x)
#         x = x * spatial_attention

#         return x

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1,
                 shortcut_type='B'):
        self.inplanes = 64
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        # self.cbam = CBAM(2048, 2048)   
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512 * block.expansion, self.num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x


def get_ResNet(model_depth = 50, num_classes=1, weights_path = 'resnet_50.pth', pretrained=True):
    '''Constructs a ResNet model
    '''
    assert model_depth in [10, 18, 34, 50, 101, 152], "model depth not in [10, 18, 34, 50, 101, 152]"

    if model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

    # get expansion
    if model_depth in [18, 34]:
        expansion = 1

    else:
        expansion = 4
  
    if pretrained == True:
        print('loading pretrained model {}'.format(weights_path))
        pretrain = torch.load(weights_path)
        model.load_state_dict(pretrain,strict=False)  # model.load_state_dict()
        print("-------- pre-train model load successfully --------")
    
    return model

############
class ResNetMLP(nn.Module):
    def __init__(self, 
                 model_depth=50,
                 num_classes=1, 
                 pretrained_path=r'E:\xyj_ccRCC_prognosis\result\feature_net\best_metric_model.pth',
                 pretrained=True,
                 freezen_weights=True):
        super(ResNetMLP, self).__init__()
        self.model_depth=model_depth
        self.num_classes=num_classes
        self.pretrained_path=pretrained_path
        self.pretrained=pretrained
        self.freezen_weights=freezen_weights

        self.expansion=4

        # Creating an instance of the ResNet-50 model
        self.resnet = get_ResNet(model_depth=self.model_depth, 
                                   num_classes=self.num_classes, 
                                   weights_path=self.pretrained_path,
                                   pretrained=self.pretrained)
        
        # Freeze all layers in the model
        if self.freezen_weights:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # move the last layer in resnet
        self.resnet.classifier=torch.nn.Identity()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512 * self.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128, num_classes))
            
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.classifier(x)
        
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0")
    # model = get_ResNet(model_depth=50, num_classes=1, pretrained=False).to(device)
    model = ResNetMLP(model_depth=50,num_classes=1,pretrained=False)
    X = torch.randn(16, 1, 224, 224, 64).to(device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Parameter {name} requires gradient and can be trained.")
    #     else:
    #         print(f"Parameter {name} does not require gradient and is fixed.")

    # for name,layer in model.named_children():
    #     X=layer(X)
    #     print("*"*10)
    #     print(name,'output shape:\t',X.shape)
    # print(X)

    # 计算参数总量
    from thop import profile
    from thop import clever_format

    input_size = (16, 1, 112, 112, 64)

    # 使用thop库进行估算
    flops, params = profile(model, inputs=(torch.randn(input_size),))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPS: {flops}")
    print(f"Total parameters: {params}")

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")


  
import torch
import torch.nn as nn
from core.utils.misc import get_logger, get_bn
device = torch.device("cuda:1")

__all__ = ['ResNet', 'resnet18_official', 'resnet34_official', 'resnet50_official', 'resnet101_official',
           'resnet152_official', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BN
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 deep_stem=False,
                 avg_down=False,
                 freeze_layer=False,
                 bn=None,
                 res_cond_bn=False):

        super(ResNet, self).__init__()

        global BN
        self.logger = get_logger(__name__)

        if norm_layer is None:
            BN = get_bn(bn)
            norm_layer = BN
        else:
            raise


        class ResCondBN(nn.Module):
            norm_layer_func = BN
            current_res = 224

            def __init__(self, *args, **kwargs):
                super(ResCondBN, self).__init__()
                self.bn112 = ResCondBN.norm_layer_func(*args, **kwargs)
                self.bn224 = ResCondBN.norm_layer_func(*args, **kwargs)

            def forward(self, input):
                if self.current_res == 224:
                    return self.bn224(input)
                elif self.current_res == 112:
                    return self.bn112(input)
                else:
                    raise NotImplementedError(f"not implemented current resolution {self.current_res}")

            def change_current_res(self, res):
                ResCondBN.current_res = res
        class ResCondBN_affine(nn.Module):
            norm_layer_func = BN
            current_res = 224

            def __init__(self, *args, **kwargs):
                super(ResCondBN_affine, self).__init__()
                self.bn112 = ResCondBN_affine.norm_layer_func(*args, **kwargs)
                self.bn224 = ResCondBN_affine.norm_layer_func(*args, **kwargs)

                assert self.bn112.num_features == self.bn224.num_features
                self.weight = nn.Parameter(torch.Tensor(self.bn112.num_features))
                self.bias = nn.Parameter(torch.Tensor(self.bn112.num_features))

                nn.init.constant_(self.weight, 1)
                nn.init.constant_(self.bias, 0)

            def forward(self, input):
                if self.current_res == 224:
                    out = self.bn224(input)
                elif self.current_res == 112:
                    out = self.bn112(input)
                else:
                    raise NotImplementedError(f"not implemented current resolution {self.current_res}")
                out = out * self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0) + self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                return out

            def change_current_res(self, res):
                ResCondBN_affine.current_res = res

        if res_cond_bn == True:
            norm_layer = ResCondBN
        elif res_cond_bn == 'affine':
            norm_layer = ResCondBN_affine

        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.num_classes = num_classes
        self.freeze_layer = freeze_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, inplanes // 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 2, inplanes // 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(inplanes // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 2, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False).to(device)

        self.bn1 = norm_layer(self.inplanes).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
        self.layer1 = self._make_layer(block, inplanes, layers[0]).to(device)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]).to(device)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]).to(device)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)) and res_cond_bn != 'affine':
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if isinstance(m.bn3, ResCondBN):
                        nn.init.constant_(m.bn3.bn112.weight, 0)
                        nn.init.constant_(m.bn3.bn224.weight, 0)
                    else:
                        nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    if isinstance(m.bn2, ResCondBN):
                        nn.init.constant_(m.bn2.bn112.weight, 0)
                        nn.init.constant_(m.bn2.bn224.weight, 0)
                    else:
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride,
                                 ceil_mode=True, count_include_pad=False),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def freeze_conv_layer(self):
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool),
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        for layer in layers:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_layer:
            self.freeze_conv_layer()
        return self

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = x.to(device)
        x = self.conv1(x)   # 0
        x = self.bn1(x)     # 1
        x = self.relu(x)    # 2
        x = self.maxpool(x) # 3

        x = self.layer1(x)  # 4
        x = self.layer2(x)  # 5
        x = self.layer3(x)  # 6
        x = self.layer4(x)  # 7

        x = self.avgpool(x) # 8
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        x = x.to(device)
        return self._forward_impl(x)


def resnet18_official(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs).to(device)
    return model


def resnet34_official(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_official(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_official(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_official(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnext50_32x4d(**kwargs):
    """
    Construct a ResNeXt-50 model with 32 groups (4 channels per group)
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnext101_32x8d(**kwargs):
    """
    Construct a ResNeXt-101 model with 32 groups (8 channels per group)
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def wide_resnet50_2(**kwargs):
    """
    Construct a Wide-ResNet-50 model with double channels
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def wide_resnet101_2(**kwargs):
    """
    Construct a Wide-ResNet-101 model with double channels
    """
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

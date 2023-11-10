import math
import torch
from torch import optim, nn
from collections import OrderedDict
from hw_layer import HWLinear, HWConv2d
import csv

def select_network(args):
    if ((args.data_type == 'mnist') or (args.data_type == 'MNIST') or 
        (args.data_type == 'cifar10') or (args.data_type == 'CIFAR10')):
            num_classes = 10
    elif (args.data_type == 'cifar100') or (args.data_type == 'CIFAR100'):
            num_classes = 100
    
    if args.network == 'plain':
        return Plain(args)
    elif 'resnet' in args.network:
        return ResNet(args, num_classes)

def get_layers():
    f = open('layers.csv', 'r', encoding='utf-8')
    layers = csv.reader(f)
    layers = list(layers)
    f.close()
    
    layers = [[param for param in layer if param != ''] for layer in layers if layer != []]

    conv_layers = []
    fc_layers = []
    resnet_blocks = []
    fc_flag = 0
    for layer in layers:
        if layer == ['fc_layers']:
            fc_flag = 1
        elif layer == ['resnet_blocks']:
            fc_flag = 2
        
        if fc_flag == 0:
            conv_layers.append(layer)
        elif fc_flag == 1:
            fc_layers.append(layer)
        else:
            resnet_blocks.append(layer)

    del conv_layers[0:2]
    del fc_layers[0:2]
    del resnet_blocks[0:2]
    
    for i in range(len(conv_layers)):
        conv_layers[i] = list(map(int, conv_layers[i]))

    for i in range(len(fc_layers)):
        fc_layers[i] = list(map(int, fc_layers[i]))

    resnet_blocks = list(map(int, resnet_blocks[0]))
    
    return conv_layers, fc_layers, resnet_blocks

def hw_conv2d(args, name, conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
              in_channels, out_channels, kernel_size, stride, padding):
    conv = HWConv2d(args, in_channels, out_channels, kernel_size, stride, padding, bias=False)
    conductances[name] = conv.conductance
    nonlinearities_LTP[name] = conv.nonlinearity_LTP
    nonlinearities_LTD[name] = conv.nonlinearity_LTD
    normalization_scales[name] = conv.normalization_scale
    
    return conv

def hw_linear(args, name, conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales, in_features, out_features):
    fc = HWLinear(args, in_features=in_features, out_features=out_features, bias=False)
    conductances[name] = fc.conductance
    nonlinearities_LTP[name] = fc.nonlinearity_LTP
    nonlinearities_LTD[name] = fc.nonlinearity_LTD
    normalization_scales[name] = fc.normalization_scale

    return fc


# Plain network
class Plain(nn.Module):
    def __init__(self, args):
        super(Plain, self).__init__()
        self.train_loss_per_batch = []
        self.train_loss_per_epoch = []
        self.test_acc_per_epoch = []
        self.conductances = OrderedDict()
        self.nonlinearities_LTP = OrderedDict()
        self.nonlinearities_LTD = OrderedDict()
        self.normalization_scales = OrderedDict()

        self.conv_config, self.fc_config, _ = get_layers()
        self.features = self.set_features_extractor(args)
        self.classifier = self.set_classifier(args)

    def set_features_extractor(self, args):
        config = self.conv_config
        layers = []
        for i, config in enumerate(config):
            conv = hw_conv2d(args, f'conv{i}', self.conductances, self.nonlinearities_LTP, self.nonlinearities_LTD, self.normalization_scales,
                             in_channels=config[0], out_channels=config[1], kernel_size=config[2], stride=config[3], padding=config[4])            
            
            if args.batch_normalization:
                layer = [conv, nn.BatchNorm2d(config[1]), nn.ReLU(inplace=True)]
            else:
                layer = [conv, nn.ReLU(inplace=True)]

            if config[5] == 1:
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif config[5] == 2:
                layer += [nn.AdaptiveAvgPool2d((1,1))]

            layers += layer

        return nn.Sequential(*layers)

    def set_classifier(self, args):
        config = self.fc_config
        
        layers = []
        for i, config in enumerate(config):
            fc = hw_linear(args, f'fc{i}', self.conductances, self.nonlinearities_LTP, self.nonlinearities_LTD, self.normalization_scales, 
                           in_features=config[0], out_features=config[1])
            layer = [fc]
            if config[2]:
                if args.batch_normalization:
                    layer += [nn.BatchNorm1d(config[1]), nn.ReLU(inplace=True)]
                else:
                    layer += [nn.ReLU(inplace=True)]
                
            layers += layer

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output


# ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, args, conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                 in_channels, out_channels, stride, block_order, block_iter):
        super().__init__()
        conv1 = hw_conv2d(args, f'BasicBlock{block_order}.{block_iter}_conv1', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                          in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        conv2 = hw_conv2d(args, f'BasicBlock{block_order}.{block_iter}_conv2', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                          in_channels=out_channels, out_channels=out_channels*BasicBlock.expansion, kernel_size=3, stride=1, padding=1)

        if args.batch_normalization:
            self.residual_function = nn.Sequential(
                conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                conv2,
                nn.BatchNorm2d(out_channels*BasicBlock.expansion),
            )
        else:
            self.residual_function = nn.Sequential(
                conv1,
                nn.ReLU(inplace=True),
                conv2,
            )
            
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            if args.batch_normalization:
                self.shortcut = nn.Sequential(
                    hw_conv2d(args, f'BasicBlock{block_order}.{block_iter}_convSC', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales, 
                            in_channels=in_channels, out_channels=out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, padding=0),
                    nn.BatchNorm2d(out_channels*BasicBlock.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    hw_conv2d(args, f'BasicBlock{block_order}.{block_iter}_convSC', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales, 
                            in_channels=in_channels, out_channels=out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, padding=0)
                )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_function(x) + self.shortcut(x)
        output = self.relu(output)
        return output


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, args, conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                 in_channels, out_channels, stride, block_order, block_iter):
        super().__init__()
        conv1 = hw_conv2d(args, f'BottleNeck{block_order}.{block_iter}_conv1', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                          in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv2 = hw_conv2d(args, f'BottleNeck{block_order}.{block_iter}_conv2', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                          in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        conv3 = hw_conv2d(args, f'BottleNeck{block_order}.{block_iter}_conv3', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales,
                          in_channels=out_channels, out_channels=out_channels*BottleNeck.expansion, kernel_size=1, stride=1, padding=0)
        
        if args.batch_normalization:
            self.residual_function = nn.Sequential(
                conv1,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                conv2,
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                conv3,
                nn.BatchNorm2d(out_channels*BottleNeck.expansion),
            )
        else:
            self.residual_function = nn.Sequential(
                conv1,
                nn.ReLU(inplace=True),
                conv2,
                nn.ReLU(inplace=True),
                conv3,
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            if args.batch_normalization:
                self.shortcut = nn.Sequential(
                    hw_conv2d(args, f'BottleNeck{block_order}.{block_iter}_convSC', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales, 
                            in_channels=in_channels, out_channels=out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, padding=0),
                    nn.BatchNorm2d(out_channels*BottleNeck.expansion),
                )
            else:
                self.shortcut = nn.Sequential(
                    hw_conv2d(args, f'BottleNeck{block_order}.{block_iter}_convSC', conductances, nonlinearities_LTP, nonlinearities_LTD, normalization_scales, 
                            in_channels=in_channels, out_channels=out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, padding=0)
                )
                
        self.relu = nn.ReLU(inplace=True)
            
    def forward(self, x):
        output = self.residual_function(x) + self.shortcut(x)
        output = self.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet, self).__init__()
        self.train_loss_per_batch = []
        self.train_loss_per_epoch = []
        self.test_acc_per_epoch = []
        self.conductances = OrderedDict()
        self.nonlinearities_LTP = OrderedDict()
        self.nonlinearities_LTD = OrderedDict()
        self.normalization_scales = OrderedDict()

        if args.network == 'resnet18':
            block = BasicBlock
            num_block = [2, 2, 2, 2]
        elif args.network == 'resnet34':
            block = BasicBlock
            num_block = [3, 4, 6, 3]
        elif args.network == 'resnet50':
            block = BottleNeck
            num_block = [3, 4, 6, 3]
        elif args.network == 'resnet101':
            block = BottleNeck
            num_block = [3, 4, 23, 3]
        elif args.network == 'resnet152':
            block = BottleNeck
            num_block = [3, 8, 36, 3]
        else:
            _, _, resnet_blocks = get_layers()
            block = BottleNeck if resnet_blocks[0] else BasicBlock
            num_block = resnet_blocks[1:]

        self.in_channels = 64
        conv = hw_conv2d(args, f'conv_block', self.conductances, self.nonlinearities_LTP, self.nonlinearities_LTD,
                         self.normalization_scales, in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=1)
        self.conv_block = nn.Sequential(
            conv,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block1 = self.set_block(args, block, 64, num_block[0], 1, 1)
        self.block2 = self.set_block(args, block, 128, num_block[1], 2, 2)
        self.block3 = self.set_block(args, block, 256, num_block[2], 2, 3)
        self.block4 = self.set_block(args, block, 512, num_block[3], 2, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc_block = hw_linear(args, f'fc_block', self.conductances, self.nonlinearities_LTP, self.nonlinearities_LTD,
                                  self.normalization_scales, in_features=512*block.expansion, out_features=num_classes)

    def set_block(self, args, block, out_channels, num_blocks, stride, block_order):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i ,stride in enumerate(strides):
            layers += [block(args, self.conductances, self.nonlinearities_LTP, self.nonlinearities_LTD, self.normalization_scales,
                             self.in_channels, out_channels, stride, block_order, i)]
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_block(x)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc_block(output)
        return output






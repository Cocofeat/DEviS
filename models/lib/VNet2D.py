import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'bn':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'gn':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'in':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'bn':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'gn':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'in':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'bn':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'gn':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'in':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'bn':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'gn':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'in':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'bn':
            ops.append(nn.BatchNorm2d(n_filters_out))
        elif normalization == 'gn':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'in':
            ops.append(nn.InstanceNorm2d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet2D(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=32, normalization='gn', has_dropout=False):
        super(VNet2D, self).__init__()

        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, n_filters * 2, normalization=normalization) # 64

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization) # 128

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization) # 256

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization) # 512

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(2, n_filters, n_filters, normalization=normalization)
        # self.block_eight_up = UpsamplingDeconvBlock(1, n_filters* 2, n_filters, normalization=normalization)
        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout2d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5 = self.block_five(x4_dw)
        # x5_dw = self.block_four_dw(x5)
        # x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x4 = self.dropout(x4)

        res = [x1, x2, x3, x4,x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        x5_up = self.block_five_up(x5) # 256
        x5_up = self.offsetAdd(x4,x5_up) # 512

        x6 = self.block_six(x5_up)  # 128
        x6_up = self.block_six_up(x6) # 128
        x6_up = self.offsetAdd(x3,x6_up) # 256

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = self.offsetAdd(x7_up , x2) # 64

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = self.offsetAdd(x8_up , x1) # 16

        x9 = self.block_nine(x8_up)
        # x10 = self.block_eight_up(x9)

        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def offsetAdd(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = down_outputs
        offset = abs(inputs.size()[3] - outputs.size()[3])
        if offset == 1:
            if outputs.device.type=='cuda':
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            else:
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3)
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            if outputs.device.type=='cuda':
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            else:
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset),
                                      out=None)
            outputs = torch.cat([outputs, addition], dim=3)
        out = inputs+outputs

        return out

    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 3, 256, 256), device=cuda0)
        # model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
        model = VNet2D(n_channels=3, n_classes=1, n_filters=32, normalization='gn', has_dropout=False)
        model.cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of model's parameter: %.2fM" % (total / 1e6))
        output = model(x)
        print('output:', output.shape)
import torch.nn as nn
import torch
from torch import autograd
from functools import partial
import torch.nn.functional as F
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)
class Unet2Ddrop(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet2Ddrop, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropoutd1 = nn.Dropout(p=0.5)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropoutd2 = nn.Dropout(p=0.5)
        self.conv5 = DoubleConv(256, 512)
        self.dropoutu1 = nn.Dropout(p=0.5)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.dropoutu2 = nn.Dropout(p=0.5)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        p3_dropout = self.dropoutd1(p3)
        c4 = self.conv4(p3_dropout)
        p4 = self.pool4(c4)
        #print(p4.shape)
        p4_dropout = self.dropoutd2(p4)
        c5 = self.conv5(p4_dropout)
        up_6 = self.up6(c5)
        up_6_dropout = self.dropoutu1(up_6)

        merge6 = self.offsetCat(c4, up_6_dropout)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        up_7_dropout = self.dropoutu1(up_7)
        merge7 = self.offsetCat(c3, up_7_dropout)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softmax()(c10)
        return out

    def offsetCat(self, inputs, down_outputs):
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
        out = torch.cat([inputs, outputs], dim=1)
        return out

class Unet2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet2D, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)

        merge6 = self.offsetCat(c4, up_6)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = self.offsetCat(c3, up_7)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softmax()(c10)
        return c10

    def offsetCat(self, inputs, down_outputs):
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
        out = torch.cat([inputs, outputs], dim=1)
        return out

class Unet2DDRIVE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet2DDRIVE, self).__init__()

        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        #print(x.shape)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        #print(p1.shape)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        #print(p2.shape)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        #print(p3.shape)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        #print(p4.shape)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)

        merge6 = self.offsetCat(c4, up_6)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = self.offsetCat(c3, up_7)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = self.offsetCat(c1, up_9)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softmax()(c10)
        return c10

    def offsetCat(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        shape1 = inputs.shape
        shape2 = down_outputs.shape
        outputs = down_outputs
        if shape1[2] != shape2[2]:
            offset = abs(inputs.size()[2] - outputs.size()[2])
            if outputs.device.type=='cuda':
                addition = torch.rand((outputs.size()[0], outputs.size()[1], offset, outputs.size()[3]), out=None).cuda()
            else:
                addition = torch.rand((outputs.size()[0], outputs.size()[1], offset, outputs.size()[3]), out=None)
            outputs = torch.cat([outputs, addition], dim=2)
        elif shape1[3] != shape2[3]:
            offset = abs(inputs.size()[3] - outputs.size()[3])
            if outputs.device.type=='cuda':
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2],offset), out=None).cuda()
            else:
                addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2],offset), out=None)
            outputs = torch.cat([outputs, addition], dim=3)
        out = torch.cat([inputs, outputs], dim=1)
        return out
nonlinearity = partial(F.relu, inplace=True)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class resnet34_unet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3,pretrained=True):
        super(resnet34_unet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center

        # Decoder
        d4 = self.downsetAdd(self.decoder4(e4), e3)
        d3 = self.decoder3(d4) + e2
        d2 = self.downsetAdd(self.decoder2(d3), e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return nn.Softmax()(out)

    def downsetAdd(self, inputs, down_outputs): # small+1, small
        # TODO: Upsampling required after deconv?
        outputs = down_outputs
        offset = abs(inputs.size()[3] - outputs.size()[3])
        if outputs.device.type=='cuda':
            out = inputs[:,:,:,0:inputs.size()[3]-offset].cuda()
        else:
            out = inputs[:,:,:,0:inputs.size()[3]-offset]

        outputs = out+down_outputs
        return outputs

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

class AttUnet2D(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttUnet2D, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Att5 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Att4 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Att2 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax()
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        offset_d5= self.offset(x4,d5)
        x4 = self.Att5(g=offset_d5, x=x4)
        d5 = torch.cat((x4, offset_d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        offset_d4 = self.offset(x3, d4)
        x3 = self.Att4(g=offset_d4, x=x3)
        d4 = torch.cat((x3, offset_d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        out = self.softmax(d1)

        return d1

    def offset(self, inputs, down_outputs):
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
        # out = torch.cat([inputs, outputs], dim=1)
        return outputs



if __name__ == '__main__':
    with torch.no_grad():
        import os
        num_classes=2
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 3, 224, 300), device=cuda0)
        # model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
        model = Unet2D(3, num_classes)
        # model = AttUnet2D(3, num_classes)
        # model = resnet34_unet(num_classes, pretrained=False)
        # model = Unet2Ddrop(3, num_classes)

        model.cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of model's parameter: %.2fM" % (total / 1e6))
        output1 = model(x)
        # output2 = model(x)
        # output3 = model(x)
        # output4 = model(x)
        print('output:', output1.shape)
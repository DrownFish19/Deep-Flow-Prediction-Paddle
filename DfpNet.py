################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.set_value(0.02 * paddle.randn(m.weight.shape))
    elif classname.find('BatchNorm') != -1:
        m.weight.set_value(paddle.normal(1.0, 0.02, shape=m.weight.shape))
        m.bias.set_value(paddle.zeros_like(m.bias))


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_sublayer('%s_relu' % name, nn.ReLU())
    else:
        block.add_sublayer('%s_leakyrelu' % name, nn.LeakyReLU(0.2))
    if not transposed:
        block.add_sublayer('%s_conv' % name, nn.Conv2D(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias_attr=True))
    else:
        block.add_sublayer('%s_upsam' % name, nn.Upsample(scale_factor=2, mode='bilinear'))  # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_sublayer('%s_tconv' % name, nn.Conv2D(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias_attr=True))
    if bn:
        block.add_sublayer('%s_bn' % name, nn.BatchNorm2D(out_c))
    if dropout > 0.:
        block.add_sublayer('%s_dropout' % name, nn.Dropout2D(dropout))
    return block


# generator model
class TurbNetG(nn.Layer):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_sublayer('layer1_conv', nn.Conv2D(3, channels, 4, 2, 1, bias_attr=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False, dropout=dropout)
        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False, dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False, dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels * 4, channels * 8, 'layer4', transposed=False, bn=True, relu=False, dropout=dropout, size=4)  # note, size 4!
        self.layer5 = blockUNet(channels * 8, channels * 8, 'layer5', transposed=False, bn=True, relu=False, dropout=dropout, size=2, pad=0)
        self.layer6 = blockUNet(channels * 8, channels * 8, 'layer6', transposed=False, bn=False, relu=False, dropout=dropout, size=2, pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels * 8, channels * 8, 'dlayer6', transposed=True, bn=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, 'dlayer5', transposed=True, bn=True, relu=True, dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True, dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True, dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_sublayer('dlayer1_relu', nn.ReLU())
        self.dlayer1.add_sublayer('dlayer1_tconv', nn.Conv2DTranspose(channels * 2, 3, 4, 2, 1, bias_attr=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b = self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = paddle.concat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = paddle.concat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = paddle.concat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = paddle.concat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = paddle.concat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = paddle.concat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Layer):
    def __init__(self, in_channels1, in_channels2, ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2D(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2D(ch, ch * 2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2D(ch * 2, ch * 4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2D(ch * 4, ch * 8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2D(ch * 8, 1, 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2D(ch * 2)
        self.bnc2 = nn.BatchNorm2D(ch * 4)
        self.bnc3 = nn.BatchNorm2D(ch * 8)

    def forward(self, x1, x2):
        h = self.c0(paddle.concat((x1, x2), 1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h)
        return h

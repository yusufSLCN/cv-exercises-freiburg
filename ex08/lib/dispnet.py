import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.blocks import deconv, conv, NegReLU
from lib.corr import Corr


def pred_block(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True),
        NegReLU(inplace=True)
    )


class DispNet(nn.Module):
    def __init__(self, C=64, corr=True):

        super().__init__()
        self.corr = corr
        input_dim = 3 if self.corr else 6

        C_curr = C  # orig: 64
        self.conv1 = conv(input_dim, C_curr, kernel_size=7, stride=2)

        C_last = C_curr  # 64
        C_curr *= 2  # 128
        self.conv2 = conv(C_last, C_curr, kernel_size=5, stride=2)

        C_last = C_curr  # 128
        C_curr *= 2  # 256
        self.conv3 = conv(C_last, C_curr, kernel_size=3, stride=2)

        if self.corr:
            self.corr_block = Corr(steps=40, step_size=1, cuda_corr=False, corr_type="disp")
            self.conv_redir = conv(C_curr, C // 2, kernel_size=1, stride=1)
            self.conv3_1 = conv(len(self.corr_block) + (C // 2), C_curr)
        else:
            self.conv3_1 = conv(C_curr, C_curr)

        C_last = C_curr  # 256
        C_curr *= 2  # 512
        self.conv4 = conv(C_last, C_curr, stride=2)
        self.conv4_1 = conv(C_curr, C_curr)

        self.conv5 = conv(C_curr, C_curr, stride=2)
        self.conv5_1 = conv(C_curr, C_curr)

        C_last = C_curr  # 512
        C_curr *= 2  # 1024
        self.conv6 = conv(C_last, C_curr, stride=2)
        self.conv6_1 = conv(C_curr, C_curr)

        self.pred_0 = pred_block(C_curr)  # input: conv6_1

        C_last = C_curr
        C_curr = C_curr // 2  # 512
        self.deconv_1 = deconv(C_last, C_curr)  # input: conv6_1
        self.pred_1 = pred_block(C_curr + C_curr + 1)  # input: [deconv_1, conv5_1, interp(pred_0)]

        C_last = C_curr  # 512
        C_curr = C_curr // 2  # 256
        self.deconv_2 = deconv(2 * C_last + 1, C_curr)  # input: [deconv_1, conv5_1, interp(pred_0)]
        self.pred_2 = pred_block(C_curr + C_last + 1)  # input: [deconv2, conv4_1, interp(pred_1)]

        C_last = C_curr  # 256
        C_curr = C_curr // 2  # 128
        self.deconv_3 = deconv(3 * C_last + 1, C_curr)  # input: [deconv2, conv4_1, interp(pred_1)]
        self.pred_3 = pred_block(C_curr + C_last + 1)  # input: [deconv3, conv3_1, interp(pred_2)]

        C_last = C_curr  # 128
        C_curr = C_curr // 2  # 64
        self.deconv_4 = deconv(3 * C_last + 1, C_curr)  # input: [deconv3, conv3_1, interp(pred_2)]
        self.pred_4 = pred_block(C_curr + C_last + 1)  # input: [deconv4, conv2a, interp(pred_3)]

        C_last = C_curr  # 64
        C_curr = int(C_curr / 2)  # 32
        self.deconv_5 = deconv(3 * C_last + 1, C_curr)  # input: [deconv4, conv2a, interp(pred_3)]
        self.pred_5 = pred_block(C_curr + C_last + 1)  # input: [deconv5, conv1a, interp(pred_4)]

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    with torch.no_grad():
                        m.bias.zero_()
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, image_left, image_right):

        if self.corr:
            conv1a, conv1b = self.conv1(image_left), self.conv1(image_right)
            conv2a, conv2b = self.conv2(conv1a), self.conv2(conv1b)
            conv3a, conv3b = self.conv3(conv2a), self.conv3(conv2b)

            corr = self.corr_block(conv3a, conv3b)

            redir = self.conv_redir(conv3a)
            merged = torch.cat([redir, corr], 1)
            conv3_1 = self.conv3_1(merged)
        else:
            images = torch.cat([image_left, image_right], 1)
            conv1a = self.conv1(images)
            conv2a = self.conv2(conv1a)
            conv3a = self.conv3(conv2a)
            conv3_1 = self.conv3_1(conv3a)

        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)

        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)

        conv6 = self.conv6(conv5_1)
        conv6_1 = self.conv6_1(conv6)

        preds = {'pred_disps_all': []}

        pred = self.pred_0(conv6_1)
        preds['pred_disps_all'].append(pred)

        deconv_1 = self.deconv_1(conv6_1)
        pred = F.interpolate(pred, size=deconv_1.shape[-2:], mode='bilinear',
                             align_corners=False).detach()
        rfeat1 = torch.cat((conv5_1, deconv_1, pred), 1)
        pred = self.pred_1(rfeat1)
        preds['pred_disps_all'].append(pred)

        deconv_2 = self.deconv_2(rfeat1)
        pred = F.interpolate(pred, size=deconv_2.shape[-2:], mode='bilinear',
                             align_corners=False).detach()
        rfeat2 = torch.cat((conv4_1, deconv_2, pred), 1)
        pred = self.pred_2(rfeat2)
        preds['pred_disps_all'].append(pred)

        deconv_3 = self.deconv_3(rfeat2)
        pred = F.interpolate(pred, size=deconv_3.shape[-2:], mode='bilinear',
                             align_corners=False).detach()
        rfeat3 = torch.cat((conv3_1, deconv_3, pred), 1)
        pred = self.pred_3(rfeat3)
        preds['pred_disps_all'].append(pred)

        deconv_4 = self.deconv_4(rfeat3)
        pred = F.interpolate(pred, size=deconv_4.shape[-2:], mode='bilinear',
                             align_corners=False).detach()
        rfeat4 = torch.cat((conv2a, deconv_4, pred), 1)
        pred = self.pred_4(rfeat4)
        preds['pred_disps_all'].append(pred)

        deconv_5 = self.deconv_5(rfeat4)
        pred = F.interpolate(pred, size=deconv_5.shape[-2:], mode='bilinear',
                             align_corners=False).detach()
        rfeat5 = torch.cat((conv1a, deconv_5, pred), 1)
        pred = self.pred_5(rfeat5)
        preds['pred_disps_all'].append(pred)

        preds['pred_disp'] = preds['pred_disps_all'][-1]
        return preds

    def visualize_corr(self, image_left, image_right, x_refs=None, y_refs=None):
        conv1a, conv1b = self.conv1(image_left), self.conv1(image_right)
        conv2a, conv2b = self.conv2(conv1a), self.conv2(conv1b)
        conv3a, conv3b = self.conv3(conv2a), self.conv3(conv2b)

        return self.corr_block.visualize(image_left, image_right, conv3a, conv3b,
                                         x_refs=x_refs, y_refs=y_refs)


class DispNetS(DispNet):
    def __init__(self, C=64):
        super().__init__(C=C, corr=False)


class DispNetC(DispNet):
    def __init__(self, C=64):
        super().__init__(C=C, corr=True)

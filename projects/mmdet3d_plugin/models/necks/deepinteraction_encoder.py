import torch
from mmcv.cnn import build_conv_layer
from torch import nn
from mmdet3d.models.builder import NECKS
from projects.mmdet3d_plugin.models.utils.encoder_utils import MMRI_I2P, LocalContextAttentionBlock, ConvBNReLU, MMRI_P2I
import pdb

class DeepInteractionEncoderLayer(nn.Module):
    def __init__(self, hidden_channel):
        super(DeepInteractionEncoderLayer, self ).__init__()
        self.I2P_block = MMRI_I2P(hidden_channel, hidden_channel, 0.1)
        self.P_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.P_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.P_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)

        self.P2I_block = MMRI_P2I(hidden_channel, hidden_channel, 9)
        self.I_IML = LocalContextAttentionBlock(hidden_channel, hidden_channel, 9)
        self.I_out_proj = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        self.I_integration = ConvBNReLU(2 * hidden_channel, hidden_channel, kernel_size = 1, norm_layer=nn.BatchNorm2d, activation_layer=None)
        
    def forward(self, img_feat, lidar_feat, img_metas, pts_metas):
        batch_size = lidar_feat.shape[0]
        BN, I_C, I_H, I_W = img_feat.shape
        I2P_feat = self.I2P_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas)
        P2P_feat = self.P_IML(lidar_feat, lidar_feat)
        P_Aug_feat = self.P_out_proj(torch.cat((I2P_feat, P2P_feat),dim=1))
        new_lidar_feat = self.P_integration(torch.cat((P_Aug_feat, lidar_feat),dim=1))

        P2I_feat = self.P2I_block(lidar_feat, img_feat.view(batch_size, -1, I_C, I_H, I_W), img_metas, pts_metas)
        I2I_feat = self.I_IML(img_feat, img_feat)
        I_Aug_feat = self.I_out_proj(torch.cat((P2I_feat.view(BN, -1, I_H, I_W), I2I_feat),dim=1))
        new_img_feat = self.I_integration(torch.cat((I_Aug_feat, img_feat),dim=1))
        return new_img_feat, new_lidar_feat

@NECKS.register_module()
class DeepInteractionEncoder(nn.Module):
    def __init__(self,
                num_layers=2,
                in_channels_img=64,
                in_channels_pts=128 * 3,
                hidden_channel=128,
                bn_momentum=0.1,
                bias='auto',
                ):
        super(DeepInteractionEncoder, self).__init__()

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.num_layers = num_layers
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.fusion_blocks.append(DeepInteractionEncoderLayer(hidden_channel))

        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas, pts_metas):
        new_img_feat = self.shared_conv_img(img_feats)
        new_pts_feat = self.shared_conv_pts(pts_feats)
        pts_feat_conv = new_pts_feat.clone()
        for i in range(self.num_layers):
            new_img_feat, new_pts_feat = self.fusion_blocks[i](new_img_feat, new_pts_feat, img_metas, pts_metas)     
        return new_img_feat, [pts_feat_conv, new_pts_feat]
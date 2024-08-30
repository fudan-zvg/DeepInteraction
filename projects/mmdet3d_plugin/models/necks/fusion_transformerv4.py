import numpy as np
import torch
from torch.nn.functional import linear
from torch.nn.init import (
    xavier_uniform_,
    constant_,
)
from einops import rearrange
from mmcv.runner import auto_fp16
from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
from mmcv.cnn import build_conv_layer
from torch import nn
from torch.nn import functional as F
from mmdet3d.models.builder import NECKS
from projects.mmdet3d_plugin.models.utils.encoder_utils import BEVWarp, LocalContextAttentionBlock, apply_3d_transformation
import copy
import math
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, ATTENTION
from mmcv.cnn.bricks.transformer import (FFN, BaseTransformerLayer, MultiScaleDeformableAttention,
                                         build_transformer_layer)
from einops import rearrange

@NECKS.register_module()
class FusionTransformerv4(nn.Module):
    def __init__(self,
                num_layers=2,
                num_lidar_maps=2,
                in_channels_img=64,
                in_channels_pts=128 * 3,
                hidden_channel=128,
                bn_momentum=0.1,
                bias='auto',
                img_transformerlayers=None,
                pts_transformerlayers=None,
                ):
        super(FusionTransformerv4, self).__init__()

        self.shared_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts * num_lidar_maps,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.multi_scale_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_img,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.multi_scale_conv_pts = build_conv_layer(
            dict(type='Conv2d'),
            in_channels_pts,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.num_layers = num_layers
        img_transformerlayers = [copy.deepcopy(img_transformerlayers) for _ in range(num_layers)]  
        self.img_fusion_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.img_fusion_blocks.append(build_transformer_layer(img_transformerlayers[i]))
        pts_transformerlayers = [copy.deepcopy(pts_transformerlayers) for _ in range(num_layers)]  
        self.pts_fusion_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.pts_fusion_blocks.append(build_transformer_layer(pts_transformerlayers[i]))
        self.bn_momentum = bn_momentum
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward(self, img_feats, pts_feats, img_metas, pts_metas):
        pts_feat_conv = self.shared_conv_pts(pts_feats.pop(0))

        ms_img_feat = [self.multi_scale_conv_img(img_feat) for img_feat in img_feats]
        ms_pts_feat = [self.multi_scale_conv_pts(pts_feat) for pts_feat in pts_feats]
        new_img_feat, new_pts_feat = ms_img_feat[0], ms_pts_feat[0]
        
        ref_img = self.get_reference_points(new_img_feat).unsqueeze(-2).repeat(
            1, 1, len(ms_img_feat), 1).to(new_img_feat.device)
        shapes_img = []
        img_feat_flatten = []
        for lvl, feat in enumerate(ms_img_feat):
            _, _, h, w = feat.shape
            feat = feat.flatten(-2).permute(0, 2, 1)
            shapes_img.append((h, w))
            img_feat_flatten.append(feat)
        img_feat_flatten = torch.cat(img_feat_flatten, -2)
        shapes_img = torch.as_tensor(
            shapes_img, dtype=torch.long, device=img_feat_flatten.device)
        start_idx_img = torch.cat((shapes_img.new_zeros(
            (1,)), shapes_img.prod(1).cumsum(0)[:-1]))

        ref_pts = self.get_reference_points(new_pts_feat).unsqueeze(-2).repeat(
            1, 1, len(ms_pts_feat), 1).to(new_pts_feat.device)
        shapes_pts = []
        pts_feat_flatten = []
        for lvl, feat in enumerate(ms_pts_feat):
            _, _, h, w = feat.shape
            feat = feat.flatten(-2).permute(0, 2, 1)
            shapes_pts.append((h, w))
            pts_feat_flatten.append(feat)
        pts_feat_flatten = torch.cat(pts_feat_flatten, -2)
        shapes_pts = torch.as_tensor(
            shapes_pts, dtype=torch.long, device=pts_feat_flatten.device)
        start_idx_pts = torch.cat((shapes_pts.new_zeros(
            (1,)), shapes_pts.prod(1).cumsum(0)[:-1]))

        for i in range(self.num_layers):
            tmp_new_img_feat = self.img_fusion_blocks[i](
                new_img_feat, new_pts_feat, img_feat_flatten, ref_img, shapes_img, start_idx_img, img_metas, pts_metas)
            tmp_new_pts_feat = self.pts_fusion_blocks[i](
                new_pts_feat, new_img_feat, pts_feat_flatten, ref_pts, shapes_pts, start_idx_pts, img_metas, pts_metas)
            new_img_feat, new_pts_feat = tmp_new_img_feat, tmp_new_pts_feat
        return new_img_feat, [pts_feat_conv, new_pts_feat]
    
    def get_reference_points(self, feat):
        H, W = feat.size()[-2:]
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H),
            torch.linspace(0.5, W - 0.5, W)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref = torch.stack((ref_x, ref_y), -1)
        return ref
        


@TRANSFORMER_LAYER.register_module()
class DeepInteractionLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 ffn_cfgs,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 **kwargs):
        super(DeepInteractionLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            batch_first=batch_first,
            **kwargs
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, query, value, ms_query, reference_points, spatial_shapes, level_start_index, img_metas, pts_metas, **kwargs):
        q_h, q_w = query.size()[-2:]
        v_h, v_w = value.size()[-2:]
        query = query.flatten(-2).permute(0, 2, 1)
        value = value.flatten(-2).reshape(-1, self.embed_dims, v_h, v_w)
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        for layer in self.operation_order[:-2]:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query=query,
                    value=ms_query,
                    identity=identity if self.pre_norm else None,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query
                self_feat = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = query.permute(0, 2, 1).reshape(-1, self.embed_dims, q_h, q_w)
                # value = value.permute(0, 2, 1).reshape(-1, self.embed_dims, v_h, v_w)
                query = self.attentions[attn_index](
                    query,
                    value,
                    img_metas=img_metas,
                    pts_metas=pts_metas,
                    reference_points=reference_points[:,:,0:1,:],
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                query = query.reshape(-1, self.embed_dims, q_h * q_w).permute(0, 2, 1)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
                
        for layer in self.operation_order[-2:]:
            if layer == 'norm':
                self_feat = self.norms[norm_index](self_feat)
                norm_index += 1
            elif layer == 'ffn':
                self_feat = self.ffns[ffn_index](self_feat)
                ffn_index += 1
        
        query = self_feat + self.scale * query
        return query.permute(0, 2, 1).reshape(-1, self.embed_dims, q_h, q_w)
        
@ATTENTION.register_module()
class MMRI_P2I(nn.Module):
    
    def __init__(self, embed_dims, batch_first=True):
        super().__init__()
        self.Warp = BEVWarp()
        self.Local = MultiScaleDeformableAttention(embed_dims, num_levels=1, batch_first=batch_first)
    
    def forward(self, img_feats, lidar_feats, img_metas, pts_metas,
            reference_points=None, **kwargs):
        B = lidar_feats.size(0)
        _, C, H, W = img_feats.size()
        warped_img_feats = self.Warp(lidar_feats, img_feats.reshape(B, -1, C, H, W), img_metas, pts_metas) #B, N, C, H, W
        img_feats = img_feats.flatten(-2).permute(0, 2, 1)
        warped_img_feats = warped_img_feats.reshape(-1,C,H*W).permute(0, 2, 1)
        shapes = torch.tensor(
            [(H, W)], device=warped_img_feats.device)
        start_idx = torch.tensor([0], device=warped_img_feats.device)
        decorated_img_feats = self.Local(query=img_feats, value=warped_img_feats, 
            reference_points=reference_points, spatial_shapes=shapes, level_start_index=start_idx)
        return decorated_img_feats.permute(0, 2, 1).reshape(-1, C, H, W)
    
@ATTENTION.register_module()
class MMRI_I2P(nn.Module):
    def __init__(self, embed_dims, dropout, batch_first=True, fp16_enabled=False, flash_attn=False, group_attn_enabled=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        if flash_attn:
            if not fp16_enabled:
                self.learnedAlign = FlashMultiheadAttention(embed_dims, 8, dropout=dropout, 
                                                        kdim=embed_dims, vdim=embed_dims, batch_first=batch_first)
            else:
                self.learnedAlign = FlashMultiheadAttention(embed_dims, 8, dropout=dropout, 
                                                        kdim=embed_dims, vdim=embed_dims, batch_first=batch_first)
        else:
            if not fp16_enabled:
                self.learnedAlign = nn.MultiheadAttention(embed_dims, 1, dropout=dropout, 
                                                    kdim=embed_dims, vdim=embed_dims, batch_first=batch_first)
            else:
                self.learnedAlign = MultiheadAttentionFP16(embed_dims, 1, dropout=dropout, 
                                                    kdim=embed_dims, vdim=embed_dims, batch_first=batch_first)
        self.group_attn_enabled = group_attn_enabled
            
    def group_attn(self, Q, K, V, attn_mask=None, groups=[20, 40, 80, 120]):
        out_tensor = Q.new_zeros(Q.shape)
        group_sum = (~attn_mask).sum(-1).squeeze(-1)
        s = 0
        for e in groups:
            group_mask = (group_sum > s) & (group_sum <= e)
            if group_mask.sum() == 0:
                s = e
                continue
            group_Q, group_K, group_V = Q[group_mask], K[group_mask], V[group_mask]
            group_attn_mask = attn_mask[group_mask]
            new_group_K = torch.cat([group_K, group_K.new_zeros(1, 1, 1).expand(group_K.shape[0], e, group_K.shape[2])], 1)
            new_group_V = torch.cat([group_V, group_V.new_zeros(1, 1, 1).expand(group_V.shape[0], e, group_V.shape[2])], 1)
            
            new_group_sum = group_sum[group_mask]
            group_padding_num = e - new_group_sum
            padding_mask = group_attn_mask.new_zeros(group_attn_mask.shape[0], e+1)
            padding_mask[torch.arange(padding_mask.shape[0],device=padding_mask.device).long(),
                         group_padding_num.long()] = 1
            padding_mask = padding_mask.cumsum(dim=1).bool().unsqueeze(1)[..., :-1]
            padded_group_mask = torch.cat([group_attn_mask, padding_mask], -1)

            new_group_K = new_group_K[~padded_group_mask.squeeze(1)].reshape(group_Q.shape[0], e, -1)
            new_group_V = new_group_V[~padded_group_mask.squeeze(1)].reshape(group_Q.shape[0], e, -1)
            
            new_groud_attn_mask = (~padding_mask).flip(-1)
            group_out = self.learnedAlign(group_Q, new_group_K, new_group_V, attn_mask=new_groud_attn_mask)[0]
            out_tensor[group_mask] = group_out
            s = e
        return out_tensor

    def forward(self, lidar_feat, img_feat, img_metas, pts_metas, **kwargs):
        B = lidar_feat.size(0)
        _, C, H, W = img_feat.size()
        img_feat = img_feat.reshape(B, -1, C, H, W)
        batch_size = len(img_metas)
        decorated_lidar_feat = torch.zeros_like(lidar_feat)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = lidar_feat.new_tensor(lidar2img) #(B,6,4,4)
        batch_cnt = lidar_feat.new_zeros(batch_size).int()
        for b in range(batch_size):
            batch_cnt[b] = (pts_metas['pillar_coors'][:,0] == b).sum()
        batch_bound = batch_cnt.cumsum(dim=0)
        cur_start = 0
        for b in range(batch_size):
            cur_end = batch_bound[b]
            voxel = pts_metas['pillars'][cur_start:cur_end]
            voxel_coor = pts_metas['pillar_coors'][cur_start:cur_end]
            pillars_num_points = pts_metas['pillars_num_points'][cur_start:cur_end]
            proj_mat = lidar2img[b]
            num_cam = proj_mat.shape[0]
            num_voxels, max_points, p_dim = voxel.shape
            num_pts = num_voxels * max_points
            pts = voxel.view(num_pts, p_dim)[...,:3]
            voxel_pts = apply_3d_transformation(pts, 'LIDAR', img_metas[b], reverse=True).detach()
            voxel_pts = torch.cat((voxel_pts,torch.ones_like(voxel_pts[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1)
            proj_mat = proj_mat.unsqueeze(1)
            xyz_cams = torch.matmul(proj_mat, voxel_pts).squeeze(-1)
            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3], torch.ones_like(xyz_cams[..., 2:3])*eps)
            img_shape = img_metas[b]['input_shape']
            xy_cams[...,0] = xy_cams[...,0] / img_shape[1]
            xy_cams[...,1] = xy_cams[...,1] / img_shape[0]
            xy_cams = (xy_cams - 0.5) * 2
            mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                 & (xy_cams[..., 0:1] < 1.0) 
                 & (xy_cams[..., 1:2] > -1.0) 
                 & (xy_cams[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask)
            sampled_feat = F.grid_sample(img_feat[b],xy_cams.unsqueeze(-2)).squeeze(-1).permute(2,0,1)
            sampled_feat = sampled_feat.reshape(num_voxels,max_points,num_cam,self.embed_dims)
            mask = mask.permute(1,0,2).reshape(num_voxels,max_points,num_cam,1)

            # for i in range(num_voxels):
            #     mask[i,pillars_num_points[i]:] = False
            mask_points = mask.new_zeros((mask.shape[0],mask.shape[1]+1))
            mask_points[torch.arange(mask.shape[0],device=mask_points.device).long(),pillars_num_points.long()] = 1
            mask_points = mask_points.cumsum(dim=1).bool()
            mask_points = ~mask_points
            mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask

            mask = mask.reshape(num_voxels,max_points*num_cam,1)
            sampled_feat = sampled_feat.reshape(num_voxels,max_points*num_cam,self.embed_dims)
            K = sampled_feat
            V = sampled_feat
            Q = lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()].t().unsqueeze(1)
            valid = mask[...,0].sum(dim=1) > 0 
            attn_output = lidar_feat.new_zeros(num_voxels, 1, self.embed_dims)
            # print(Q[valid].shape, K[valid].shape, V[valid].shape, (~mask[valid]).permute(0,2,1).shape)
            if self.group_attn_enabled:
                attn_output[valid] = self.group_attn(Q[valid],K[valid],V[valid],attn_mask=(~mask[valid]).permute(0,2,1))
            else:
                attn_output[valid] = self.learnedAlign(Q[valid],K[valid],V[valid],attn_mask=(~mask[valid]).permute(0,2,1))[0]
            decorated_lidar_feat[b,:,voxel_coor[:,2].long(),voxel_coor[:,3].long()] = attn_output.squeeze(1).t()
            cur_start = cur_end
        return decorated_lidar_feat + lidar_feat


class TransSinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(TransSinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, x_range, y_range, z_pos=None):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        bs = x_range.shape[0]
        if len(x_range.shape) <=2:
            x_len = x_range.shape[-1]
            y_len = y_range.shape[-1]
            x_embed = x_range.unsqueeze(-2).repeat(1, y_len, 1)
            y_embed = y_range.unsqueeze(-1).repeat(1, 1, x_len)
        else:
            y_len, x_len = x_range.shape[1:]
            x_embed = x_range
            y_embed = y_range

        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale

        if z_pos:
            z_embed = z_pos.view(bs,1,1).repeat(1, y_len, x_len)
            num_feat_xy = self.num_feats-2
            num_feat_z = 4
            
            dim_t_xy = torch.arange(
                num_feat_xy, dtype=torch.float32, device=x_range.device)
            dim_t_z = torch.arange(
                num_feat_z, dtype=torch.float32, device=x_range.device)
            dim_t_xy = self.temperature**(2 * (dim_t_xy // 2) / num_feat_xy)
            dim_t_z = self.temperature**(2 * (dim_t_z // 2) / num_feat_z)

            pos_x = x_embed[:, :, :, None] / dim_t_xy
            pos_y = y_embed[:, :, :, None] / dim_t_xy
            pos_z = z_embed[:, :, :, None] / dim_t_z
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            B, H, W = bs, y_len, x_len
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_z = torch.stack(
                (pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)
        else:
            dim_t = torch.arange(
                self.num_feats, dtype=torch.float32, device=x_range.device)
            dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            B, H, W = bs, y_len, x_len
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
@ATTENTION.register_module()
class MMRI_I2P_Polar(nn.Module):
    def __init__(self, embed_dims, dropout, batch_first=True, radius_range=[1., 61., 1.0], num_decoder_layers=1, pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]):
        super().__init__()
        self.embed_dims = embed_dims
        self.dropout = dropout
        
        self.pos_encoding = TransSinePositionalEncoding(int(embed_dims/2))
        self.radius_range = radius_range
        self.radius = int((radius_range[1] - radius_range[0])/radius_range[-1])
        self.im_scale = 4.
        self.pc_range = pc_range
        
        decoder_layer = FlashTransformerDecoderLayer(d_model=embed_dims, nhead=8, dim_feedforward=embed_dims*4, batch_first=True)
        decoder_norm = nn.LayerNorm(embed_dims)
        decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.transformer_layers = nn.Transformer(d_model=embed_dims, nhead=8, dim_feedforward=embed_dims*4,
                                        num_encoder_layers=0, num_decoder_layers=1, batch_first=True, custom_decoder=decoder)

    def forward(self, lidar_feat, img_feat, img_metas, pts_metas, **kwargs):
        B = lidar_feat.size(0)
        _, C, H, W = img_feat.size()
        R = self.radius
        img_feat = img_feat.reshape(B, -1, C, H, W)
        batch_size = len(img_metas)
        decorated_lidar_feat = torch.zeros_like(lidar_feat)
        visibles = torch.zeros_like(decorated_lidar_feat[:,0:1])
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = lidar_feat.new_tensor(lidar2img) #(B,6,4,4)
        
        cam_intrinsic = []
        for img_meta in img_metas:
            cam_intrinsic.append(img_meta['cam_intrinsic'])
        cam_intrinsic = np.asarray(cam_intrinsic)
        cam_intrinsic = lidar_feat.new_tensor(cam_intrinsic)
        
        cam2lidar = []
        for img_meta in img_metas:
            cam2lidar.append(img_meta['cam2lidar'])
        cam2lidar = np.asarray(cam2lidar)
        cam2lidar = lidar_feat.new_tensor(cam2lidar)
        
        ###################################################
        num_cam = img_feat.shape[1]
        
        for cam_id in range(num_cam):
            feature_single_cam = img_feat[:,cam_id]
            
            img_x_range = torch.arange(0., float(W), 1., device=feature_single_cam.device)
            img_x_range = img_x_range.unsqueeze(0).repeat(B, 1)
            img_y_range = torch.arange(0., float(H), 1., device=feature_single_cam.device)
            img_y_range = img_y_range.unsqueeze(0).repeat(B, 1)
            img_pos = self.pos_encoding(img_x_range, img_y_range) # B, C, H, W
            
            polar_x_range = torch.arange(0., float(W), 1., device=feature_single_cam.device)
            polar_x_range = polar_x_range.unsqueeze(0).repeat(B, 1)
            polar_y_range = torch.arange(0., float(R), 1., device=feature_single_cam.device)
            polar_y_range = polar_y_range.unsqueeze(0).repeat(B, 1)
            polar_rays_pos = self.pos_encoding(polar_x_range, polar_y_range) # [B, C*2, R, W]
            
            cam_coors = torch.stack([img_x_range+0.5, torch.zeros_like(img_x_range)+H//2, 
                                     torch.ones_like(img_x_range), torch.ones_like(img_x_range)], dim=1)
            cam_coors[:,:2] *= self.im_scale
            
            img2lidar = torch.linalg.inv(lidar2img[:,cam_id])
            cam_coors_lidar = torch.bmm(img2lidar, cam_coors)[:,:2]
            cam_coords = cam2lidar[:, cam_id, :2, -1:]
            ray_dirs = cam_coors_lidar - cam_coords
            ray_dirs = ray_dirs / ray_dirs.norm(dim=1, p=2, keepdim=True)
            
            depths = torch.arange(self.radius_range[0], self.radius_range[1], self.radius_range[2]) + self.radius_range[2]/2
            polar_centers = depths[None,None,:,None].to(ray_dirs) * ray_dirs[:,:,None]
            polar_centers = rearrange(polar_centers, 'b c r w -> b r w c')
            
            polar_centers_aug = []
            for b in range(B):
                coor_batch = polar_centers[b]
                coor_batch = rearrange(coor_batch, 'r w c -> (r w) c')
                coor_batch = torch.cat([coor_batch, torch.zeros_like(coor_batch[:,:1])], -1)
                coor_batch_aug = apply_3d_transformation(coor_batch, 'LIDAR', img_metas[b], reverse=False).detach()
                coor_batch_aug = rearrange(coor_batch_aug, '(r w) c -> r w c', r=R, w=W)
                polar_centers_aug.append(coor_batch_aug)
                
            polar_centers_aug = torch.stack(polar_centers_aug, dim=0)
            polar_centers_aug_norm = torch.zeros_like(polar_centers_aug[...,:2])
            polar_centers_aug_norm[...,0] = (polar_centers_aug[...,0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
            polar_centers_aug_norm[...,1] = (polar_centers_aug[...,1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
            
            polar_query = F.grid_sample(lidar_feat, polar_centers_aug_norm*2-1, align_corners=False)
            polar_query = polar_query + polar_rays_pos
            polar_rays = polar_query.permute(2, 0, 3, 1).flatten(1, 2) # [R, B*W, C*2]
            
            img_columns = feature_single_cam + img_pos
            img_columns = img_columns.permute(2, 0, 3, 1).flatten(1, 2) # [H, B*W, C]
            
            bev_out = self.transformer_layers(img_columns.transpose(0, 1), polar_rays.transpose(0, 1))
            bev_out = bev_out.view(B, W, R, C).permute(0, 3, 2, 1) # [B, C, R, W]
            # bev_out = self.transformer_layers(img_columns, polar_rays)
            # bev_out = bev_out.view(R, B, W, C).permute(1, 3, 0, 2) # [B, C, R, W]
            
            x_size, y_size, z_size = lidar_feat.shape[-2], lidar_feat.shape[-1], 10
            meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size], [0, z_size - 1, z_size]]
            batch_y, batch_x, batch_z = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) + 0.5 for it in meshgrid])
            batch_x = batch_x / x_size * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            batch_y = batch_y / y_size * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            batch_z = batch_z / z_size * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            bev_pts = torch.stack([batch_x, batch_y, batch_z], -1)[None].repeat(B,1,1,1,1)
            bev_pts_reaugs = []
            for b in range(B):
                bev_pts_batch = bev_pts[b]
                bev_pts_batch = rearrange(bev_pts_batch, 'w h z c -> (w h z) c')
                bev_pts_reaug = apply_3d_transformation(bev_pts_batch, 'LIDAR', img_metas[b], reverse=True).detach()
                bev_pts_reaugs.append(bev_pts_reaug)
            
            bev_pts_reaugs = torch.stack(bev_pts_reaugs, 0)
            proj_mat = lidar2img[:,cam_id]
            bev_pts_reaugs = torch.cat((bev_pts_reaugs,torch.ones_like(bev_pts_reaugs[...,:1])),dim=-1).transpose(1,2).to(proj_mat.device)
            xyz_cams = torch.bmm(proj_mat, bev_pts_reaugs)[:,:3].transpose(1,2)
            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3], torch.ones_like(xyz_cams[..., 2:3])*eps)
            img_shape = img_metas[0]['input_shape']
            xy_cams[...,0] = xy_cams[...,0] / img_shape[1]
            xy_cams[...,1] = xy_cams[...,1] / img_shape[0]
            xy_cams = 2 * xy_cams - 1
            mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                 & (xy_cams[..., 0:1] < 1.0) 
                 & (xy_cams[..., 1:2] > -1.0) 
                 & (xy_cams[..., 1:2] < 1.0))
            sampling_pixel = xy_cams[..., 0] # -1 ~ 1

            grid_map = bev_pts_reaugs[:,:2,:] - cam_coords
            radius_map = torch.norm(grid_map, dim=1)
            norm_radius_map = (2*(radius_map-self.radius_range[0])/self.radius -1).clamp(-1, 1)
            sample_loc = torch.stack([sampling_pixel, norm_radius_map], -1)
            sample_loc = rearrange(sample_loc, 'b (w h z) c -> b w h z c', w=y_size, h=x_size, z=z_size).mean(dim=3)
            mask = rearrange(mask, 'b (w h z) c -> b w h z c', w=y_size, h=x_size, z=z_size).sum(dim=3).permute(0,3,1,2) > 0
            
            sampling = F.grid_sample(bev_out, sample_loc)
            
            # mask[:,0] = mask[:,0] & (sample_loc[...,0] > -1.0) & (sample_loc[...,0] < 1.0) & (sample_loc[...,1] > -1.0) & (sample_loc[...,1] < -1.0)
            
            decorated_lidar_feat += sampling * mask
            visibles += mask
            
        visibles[visibles == 0] = 1
        decorated_lidar_feat = decorated_lidar_feat / visibles
        
        return decorated_lidar_feat + lidar_feat


def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k 
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                    device=kv.device)                    
            output = flash_attn_unpadded_kvpacked_func(
                q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, batch_first=True,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
    
    def forward(self, q, k, v, key_padding_mask=None, need_weights=None, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


class FlashTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=False, **kwargs):
        super().__init__(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=batch_first, **kwargs)
        del self.self_attn
        del self.multihead_attn
        self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

    
class MultiheadAttentionFP16(nn.MultiheadAttention):
    @auto_fp16(apply_to=('q', 'k', 'v'), out_fp32=True)
    def forward(self, q, k, v, **kwargs):
        return super(MultiheadAttentionFP16, self).forward(q, k, v, **kwargs)
        
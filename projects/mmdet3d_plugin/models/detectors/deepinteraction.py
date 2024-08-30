import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
# from mmdet3d.ops import Voxelization
from ..updated_modules import SPConvVoxelization as Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class DeepInteraction(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 freeze_img=False,
                 freeze_pts=False,
                 multi_scale=False,
                 pts_pillar_layer=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 imgpts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DeepInteraction, self).__init__(None, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_pillar_layer is not None:
            self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        if imgpts_neck is not None:
            self.imgpts_neck = builder.build_neck(imgpts_neck)
        self.multi_scale = multi_scale
        self.freeze_img = freeze_img
        self.freeze_pts = freeze_pts
        
    def init_weights(self):
        """Initialize model weights."""
        super(DeepInteraction, self).init_weights()
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
        
        if self.freeze_pts:
            for name, param in self.named_parameters():
                if 'pts' in name and 'pts_bbox_head' not in name and 'imgpts_neck' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.decoder.0' in name:
                    param.requires_grad = False
                if 'imgpts_neck.shared_conv_pts' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                    param.requires_grad = False
                if 'pts_bbox_head.prediction_heads.0' in name:
                    param.requires_grad = False
                if 'pts_bbox_head.class_encoding' in name:
                    param.requires_grad = False
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
            self.pts_voxel_layer.apply(fix_bn)
            self.pts_voxel_encoder.apply(fix_bn)
            self.pts_middle_encoder.apply(fix_bn)
            self.pts_backbone.apply(fix_bn)
            self.pts_neck.apply(fix_bn)
            self.pts_bbox_head.heatmap_head.apply(fix_bn)
            self.pts_bbox_head.class_encoding.apply(fix_bn)
            self.pts_bbox_head.decoder[0].apply(fix_bn)
            self.pts_bbox_head.prediction_heads[0].apply(fix_bn)            
            self.imgpts_neck.shared_conv_pts.apply(fix_bn)


    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts,voxel_type='voxel')
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        
        pillars, pillars_num_points, pillar_coors = self.voxelize(pts, voxel_type='pillar')
        pillar_features = self.pts_voxel_encoder(pillars, pillars_num_points, pillar_coors)
        pts_metas = {}
        pts_metas['pillar_center'] = pillar_features
        pts_metas['pillars'] = pillars
        pts_metas['pillars_num_points'] = pillars_num_points
        pts_metas['pillar_coors'] = pillar_coors
        pts_metas['pts'] = pts
        return x, pts_metas
    
    def extract_feat(self, points, img, img_metas):
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_metas = self.extract_pts_feat(points, img_feats, img_metas)
        if not self.multi_scale:
            new_img_feat, new_pts_feat = self.imgpts_neck(img_feats[0], pts_feats[0], img_metas, pts_metas)
        else:
            new_img_feat, new_pts_feat = self.imgpts_neck(img_feats[:2], pts_feats, img_metas, pts_metas)
        return (new_img_feat, new_pts_feat)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, voxel_type='voxel'):
        assert voxel_type=='voxel' or voxel_type=='pillar'
        voxels, coors, num_points = [], [], []
        for res in points:
            if voxel_type == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            elif voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feats, img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

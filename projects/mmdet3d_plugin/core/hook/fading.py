import enum
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Fading(Hook):
    def __init__(self, fade_epoch = 100000):
        """ Freeze Layers Hook.
        
        Args:
            freeze_layers (list[str]): Names of frozed layers, e.g. "img_backbone.layer1".
            aug_layers (list[str]): Names of augmentation layers added in the model.
                Aug_layers shouldn't be frozen.
        """
        self.fade_epoch = fade_epoch

    def before_train_epoch(self, runner):
        if runner.epoch == self.fade_epoch:
            for i, transform in enumerate(runner.data_loader.dataset.dataset.pipeline.transforms):
                if type(transform).__name__ == 'ObjectSample':
                    runner.data_loader.dataset.dataset.pipeline.transforms.pop(i)
                    break
# DeepInteraction: 3D Object Detection via Modality Interaction
### [Paper](https://arxiv.org/abs/2208.11112)
> [**DeepInteraction: 3D Object Detection via Modality Interaction**](https://arxiv.org/abs/2208.11112),            
> Zeyu Yang, Jiaqi Chen, Zhenwei Miao, [Wei Li](https://weivision.github.io/), [Xiatian Zhu](https://xiatian-zhu.github.io), [Li Zhang](https://www.robots.ox.ac.uk/~lz)

## News

- **(2022/6/27)** DeepInteraction-e ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions.
- **(2022/6/26)** DeepInteraction-large ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all non-ensemble solutions.
- **(2022/5/18)** DeepInteraction-base ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions that do not use test-time augmentation and model ensemble.


## Results

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| DeepInteraction-e |   C+L    | 75.74 | 76.34 |
| DeepInteraction-large |   C+L    | 74.12 | 75.52 |
| DeepInteraction-base |   C+L    | 70.78 | 73.43 |

### 3D Object Detection (on nuScenes val)

|   Model   | Modality | mAP  | NDS  | Checkpoint |
| :-------: | :------: | :--: | :--: | :--------: |
| [DeepInteraction-base](projects/configs/nuscenes/Fusion_0075_refactor.py) |   C+L    | 69.85 | 72.63 | [Fusion_0075_refactor.pth](https://drive.google.com/file/d/1M5eUlXZ8HJ--J53y0FoAHn1QpZGowsdc/view?usp=sharing) |

## Get Started

### Environment
This implementation is build upon [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), please follow the steps in [install.md](./install.md) to prepare the environment.

### Data
Please follow the official instructions of mmdetection3d to process the nuScenes dataset.(https://mmdetection3d.readthedocs.io/en/latest/datasets/nuscenes_det.html)

### Pretrained
Downloads the [pretrained backbone weights](https://drive.google.com/file/d/1IaLMcRu4SYTqcD6K1HF5UjfnRICB_IQM/view?usp=sharing) to pretrained/ 

### Train & Test
```shell
# train with 8 GPUs
tools/dist_train.sh projects/configs/nuscenes/Fusion_0075_refactor.py 8
# test with 8 GPUs
tools/dist_test.sh projects/configs/nuscenes/Fusion_0075_refactor.py ${CHECKPOINT_FILE} 8 --eval=bbox
```

## Acknowledgement
Many thanks to the following open-source projects:
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* [detectron2](https://github.com/facebookresearch/detectron2)  
* [transfusion](https://github.com/XuyangBai/TransFusion)


## Reference

```bibtex
@inproceedings{yang2022deepinteraction,
  title={DeepInteraction: 3D Object Detection via Modality Interaction},
  author={Yang, Zeyu and Chen, Jiaqi and Miao, Zhenwei and Li, Wei and Zhu, Xiatian and Zhang, Li},
  booktitle={NeurIPS},
  year={2022}
}
```

# DeepInteraction: Exploring Multi-modal Interaction for 3D Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/bevfusion-multi-task-multi-sensor-fusion-with/3d-object-detection-on-nuscenes)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes?p=bevfusion-multi-task-multi-sensor-fusion-with)

## News

- **(2022/6/27)** DeepInteraction-e ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions.
- **(2022/6/27)** DeepInteraction-large ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all non-ensemble solutions.
- **(2022/5/18)** DeepInteraction-base ranks first on [nuScenes](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Any) among all solutions that do not use test-time augmentation and model ensemble.


## Results

### 3D Object Detection (on nuScenes test)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| BEVFusion-e |   C+L    | 75.62 | 76.28 |
| BEVFusion-large |   C+L    | 74.12 | 75.52 |
| BEVFusion-base |   C+L    | 70.78 | 73.43 |

### 3D Object Detection (on nuScenes val)

|   Model   | Modality | mAP  | NDS  |
| :-------: | :------: | :--: | :--: |
| BEVFusion-base |   C+L    | 69.85 | 72.63 |


## Get Started

### Environment
This implementation is build upon [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), please follow the steps in [install.md](./docs/install.md) to prepare the environment.

### Data
Please follow the official instructions of mmdetection3d to process the nuScenes dataset.(https://mmdetection3d.readthedocs.io/en/latest/datasets/nuscenes_det.html)


## Acknowledgement
Many thanks to the following open-source projects:
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* [detectron2](https://github.com/facebookresearch/detectron2)  
* [transfusion](https://github.com/XuyangBai/TransFusion)


## Reference

```bibtex
@article{yang2022deepinteraction,
  title={DeepInteraction: Exploring Multi-modal Interaction for 3D Object Detection},
  author={Yang, Zeyu and Chen, Jiaqi and Miao, Zhenwei and Li, Wei and Zhu, Xiatian and Zhang, Li},
  journal={arXiv},
  year={2022}
}
```

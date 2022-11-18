# Installation
## Prerequisite
1. `Python=3.7`
2. `Pytorch=1.9.1`
3. `CUDA=11.1`

### **Step. 1** Install mmcv-full==1.4.1. 
```
pip install mmcv-full==1.3.18
```

### **Step. 2** Install mmdetection and mmsegmentation. 
```
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

### **Step. 3** Install mmdetection3d==0.17.1. 
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```

### **Step. 4** Install detectron2. 
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

### **Step. 5** Clone deepInteraction. 
```
git clone https://github.com/fudan-zvg/DeepInteraction.git
cd DeepInteraction
```
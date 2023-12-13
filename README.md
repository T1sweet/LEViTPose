# Lightweight Human Pose Estimation with Efficient Vision Transformer

## Code coming soon...

## Paper

> [(英文)Lightweight Human Pose Estimation with Efficient Vision Transformer](https://arxiv.org/pdf/2311.13615.pdf)  
> Chengpeng Wu, Guangxing Tan*, Haifeng Chen, Chunyu Li

> [(中文)融入高效视觉变换器的轻量级人体姿态估计, 计算机工程与应用](https://arxiv.org/pdf/2311.13615.pdf)  
> 吴程鹏, 谭光兴*, 陈海峰, 李春宇

## The network architecture of HEVITPose

![overview](img/NetworkGraph.png)

## Main Results
With the code contained in this repo, you should be able to reproduce the following results. 
### Results on MPII val and test set
|   Method      |   Test set    | Input size |Params |GFLOPs | Hea| Sho| Elb| Wri |Hip| Kne |Ank |mean|
|---------------|---------------|------------|-------|-------|----|----|----|-----|----|-----|----|-----|
| LiteHEViTPose-T   | MPII val      |  256×256   | 1.09M | 0.89G | 95.6	|93.8 |86.3	|79.9 |86.3	|79.9 |74.5	|85.9|
| LiteHEViTPose-S   | MPII val      |  256×256   | 2.16M | 1.33G | 96.1	|95.0 |87.9	|81.9 |87.8	|82.6 |77.7	|87.7|
| LiteHEViTPose-T   | MPII test     |  256×256   | 1.09M | 0.89G | 97.5	|94.6 |88.2	|82.1 |88.0	|82.2 |76.7	|87.6|
| LiteHEViTPose-S   | MPII test     |  256×256   | 2.16M | 1.33G | 97.8	|95.4 |89.6	|84.1 |89.1	|84.0 |79.8	|89.0|

### Results on COCO val2017 and test-dev2017 set
| Method     | Test set      | Input size |  AP | AP .5|AP .75|AP (M)|AP (L)| AR   |
|------------|---------------|------------|-----|------|------|------|------|------| 
| LiteHEViTPose-S| COCO val      | 256×256    | 71.0| 91.6 | 78.5 |	68.2 | 75.1 | 74.1|
| LiteHEViTPose-S| COCO test-dev | 256×256    | 68.7| 90.8 | 76.7 |	65.4 | 74.2 | 74.4|

## Visualization
Some examples of the prediction results of the LiteHEViTPose network model for
human posture include occlusion, multiple people, viewpoint and appearance change on the MPII (top) and COCO (bottom) data sets.

![Visualization](./img/visualization.png)



## Installation

### 1. Clone code
```shell
    git clone https://github.com/T1sweet/LiteHEViTPose
    cd ./LiteHEViTPose
```

### 2. Create a conda environment for this repo
```shell
    conda create -n LiteHEViTPose python=3.9
    conda activate LiteHEViTPose
```

### 3. Install PyTorch >= 1.6.0 following official instruction, e.g.
Our model is trained in a GPU platforms and relies on the following versions: 
torch==1.10.1+cu113, torchvision==0.11.2+cu113
```shell
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

### 4. Install other dependency python packages
Our code is based on the MMPose 0.29.0 code database, and dependencies can be installed through the methods provided by [MMPose](https://github.com/open-mmlab/mmpose/blob/v0.29.0/docs/en/install.md). 
Install MMCV using MIM.
```shell
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    pip install -U openmim
    mim install mmcv-full==1.4.5
```
Install other dependency.
```shell
    pip install -r requirements.txt
```

### 5. Prepare dataset
Download [MPII](http://human-pose.mpi-inf.mpg.de/#download) and [COCO ](https://cocodataset.org/#home) from website and put the zip file under the directory following below structure, (xxx.json) denotes their original name.

```
./data
|── coco
│   └── annotations
|   |   └──coco_train.json(person_keypoints_train2017.json)
|   |   └──coco_val.json(person_keypoints_val2017.json)
|   |   └──coco_test.json(image_info_test-dev2017.json)
|   └── images
|   |   └──train2017
|   |   |   └──000000000009.jpg
|   |   └──val2017
|   |   |   └──000000000139.jpg
|   |   └──test2017
|   |   |   └──000000000001.jpg
├── mpii
│   └── annotations
|   |   └──mpii_train.json(refer to DEKR, link:https://github.com/HRNet/DEKR)
|   |   └──mpii_val.json
|   |   └──mpii_test.json
|   |   └──mpii_gt_val.mat
|   └── images
|   |   └──100000.jpg
```
## Usage

### 1. Download trained model
* [MPII](https://1drv.ms/u/s!AhpKYLhXKpH7gv8RepyMU_iU5uhxhg?e=ygs4Me)
* [COCO](https://1drv.ms/u/s!AhpKYLhXKpH7gv8RepyMU_iU5uhxhg?e=ygs4Me)


### 2. Evaluate Model
Change the checkpoint path by modifying `pretrained` in HEViTPose-B_mpii_256x256.py, and run following commands:
python tools/test.py config checkpoint
`config` option means the configuration file, which must be set.
`checkpoint` option means the training weight file and must be set.

```python
# evaluate LiteHEViTPose-S on mpii val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LiteHEViTPose-S_mpii_256x256.py /work_dir/LiteHEViTPose/LiteHEViTPose-S.pth

# evaluate LiteHEViTPose-T on mpii val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LiteHEViTPose-T_mpii_256x256.py /work_dir/LiteHEViTPose/LiteHEViTPose-T.pth

# evaluate LiteHEViTPose-S on coco val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LiteHEViTPose-B_coco_256x256.py /work_dir/LiteHEViTPose/LiteHEViTPose-B_coco.pth

```

### 3. Train Model
Change the checkpoint path by modifying `pretrained` in LiteHEViTPose-B_mpii_256x256.py, and run following commands:
```python
# evaluate LiteHEViTPose-S on mpii val set
python tools/train.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LiteHEViTPose-B_mpii_256x256.py

# evaluate LiteHEViTPose-S on coco val2017 set
python tools/train.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LiteHEViTPose-B_coco_256x256.py
```

## Contact me
If you have any questions about this code or paper, feel free to contact me at
CP935011539@outlook.com.


## Citations
If you find this code useful for your research, please cite our paper:

```bibtex
@misc{wu2024litehevitpose,
    title     = {Lightweight Human Pose Estimation with Efficient Vision Transformer},
    author    = {Chengpeng Wu, Guangxing Tan*, Haifeng Chen, Chunyu Li},

}
```


## Acknowledgement
This algorithm is based on code database [MMPose](https://github.com/open-mmlab/mmpose/tree/v0.29.0), and its main ideas are inspired by [EfficientViT](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.pdf)) and other papers.

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

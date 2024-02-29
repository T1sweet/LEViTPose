# Lightweight and Efficient Human Pose Estimation

## Code coming soon...

## Paper
> Chengpeng Wu, Guangxing Tan, Haifeng Chen, Chunyu Li. Lightweight and Efficient Human Pose Estimation Fusing Transformer and Attention[J]. Computer Engineering and Applications. 2024.
> 吴程鹏, 谭光兴, 陈海峰, 李春宇. 融合Transformer和注意力的轻量高效人体姿态估计[J]. 计算机工程与应用. 2024.

## The network architecture of HEVITPose

![overview](img/NetworkGraph.png)

## Main Results
With the code contained in this repo, you should be able to reproduce the following results. 
### Results on MPII val and test set
|   Method      |   Test set    | Input size |Params |GFLOPs | Hea| Sho| Elb| Wri |Hip| Kne |Ank |mean|
|---------------|---------------|------------|-------|-------|----|----|----|-----|----|-----|----|-----|
| LEViTPose-T   | MPII val      |  256×256   | 1.09M | 0.89G | 95.6	|93.8 |86.3	|79.9 |86.3	|79.9 |74.5	|85.9|
| LEViTPose-S   | MPII val      |  256×256   | 2.16M | 1.33G | 96.1	|95.0 |87.9	|81.9 |87.8	|82.6 |77.7	|87.7|
| LEViTPose-T   | MPII test     |  256×256   | 1.09M | 0.89G | 97.5	|94.6 |88.2	|82.1 |88.0	|82.2 |76.7	|87.6|
| LEViTPose-S   | MPII test     |  256×256   | 2.16M | 1.33G | 97.8	|95.4 |89.6	|84.1 |89.1	|84.0 |79.8	|89.0|

### Results on COCO val2017 and test-dev2017 set
| Method     | Test set      | Input size |  AP | AP .5|AP .75|AP (M)|AP (L)| AR   |
|------------|---------------|------------|-----|------|------|------|------|------| 
| LEViTPose-S| COCO val      | 256×256    | 71.0| 91.6 | 78.5 |	68.2 | 75.1 | 74.1|
| LEViTPose-T| COCO val      | 256×256    | 68.2 | 90.5 | 76.0 | 65.6 | 72.2| 71.5|
| LEViTPose-S| COCO test-dev | 256×256    | 68.7| 90.8 | 76.7 |	65.4 | 74.2 | 74.4|

### Comparison of inference speed of models in MPII Dataset 
| Method          | Params | FLOPs  | FPS(GPU) | FPS(CPU) | mean |
|-----------------|--------|--------|----------|----------|------|
| HRNet-W32       | 28.02M | 9.85G  |   31.9   |    2.5   | 89.6 | 
| Hourglass-52    | 94.85M | 28.67G |   25.7   |    1.3   | 88.9 | 
| EfficientViT-M0 | 3.04M  | 1.89G  |   52.5   |    5.7   | 85.8 | 
| LiteHRNet-30    | 1.76M  | 0.56G  |   29.9   |    4.5   | 85.1 | 
| MobileNetV2     | 9.57M  | 2.12G  |   67.9   |    6.2   | 85.0 | 
| PVT-S           | 28.17M | 5.47G  |   31.7   |    2.6   | 84.4 | 
| LEViTPose-S     | 2.16M  | 1.45G  |   55.0   |    6.4   | 87.7 | 
| LEViTPose-T     | 1.09M  | 0.89G  |   60.1   |    7.5   | 85.9 | 

## Visualization
Some examples of the prediction results of the LEViTPose network model for
human posture include occlusion, multiple people, viewpoint and appearance change on the MPII (top) and COCO (bottom) data sets.

![Visualization](./img/visualization.png)


## Installation

### 1. Clone code
```shell
    git clone https://github.com/T1sweet/LEViTPose
    cd ./LEViTPose
```

### 2. Create a conda environment for this repo
```shell
    conda create -n LEViTPose python=3.9
    conda activate LEViTPose
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
# evaluate LEViTPose-S on mpii val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LEViTPose-S_mpii_256x256.py /work_dir/LEViTPose/LEViTPose-S.pth

# evaluate LEViTPose-T on mpii val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LEViTPose-T_mpii_256x256.py /work_dir/LEViTPose/LEViTPose-T.pth

# evaluate LEViTPose-S on coco val set
python tools/test.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LEViTPose-B_coco_256x256.py /work_dir/LEViTPose/LEViTPose-B_coco.pth

```

### 3. Train Model
Change the checkpoint path by modifying `pretrained` in LEViTPose-B_mpii_256x256.py, and run following commands:
```python
# evaluate LEViTPose-S on mpii val set
python tools/train.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LEViTPose-B_mpii_256x256.py

# evaluate LEViTPose-S on coco val2017 set
python tools/train.py ../configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/LEViTPose-B_coco_256x256.py
```

## Contact me
If you have any questions about this code or paper, feel free to contact me at
CP935011539@outlook.com.


## Citations
If you find this code useful for your research, please cite our paper:

```bibtex
@misc{wu2024LEViTPose,
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

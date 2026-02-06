# Shape Generation and Completion Through Point-Voxel Diffusion
<p align="center">
  <img src="assets/pvd_teaser.gif" width="80%"/>
</p>

[Project](https://alexzhou907.github.io/pvd) | [Paper](https://arxiv.org/abs/2104.03670) 

Implementation of Shape Generation and Completion Through Point-Voxel Diffusion

[Linqi Zhou](https://alexzhou907.github.io), [Yilun Du](https://yilundu.github.io/), [Jiajun Wu](https://jiajunwu.com/)

## Requirements:

Make sure the following environments are installed.

```
python>=3.11
pytorch>=2.8.0
torchvision>=0.23.0
cuda>=12.8
matplotlib>=3.4.0
tqdm>=4.60.0
open3d>=0.18.0
trimesh>=4.0.0
scipy>=1.10.0
tensorboard>=2.15.0
h5py>=3.0.0
```

```
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric  -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

Install dependencies using:
```bash
pip install -r requirement_voxel.txt
```

The code was tested on Ubuntu with Nvidia RTX 5090. 

## Data

For NUP96 point cloud generation, we use preprocessed H5 files generated from SMLM data.

Data preprocessing pipeline:
1. Run data cleaning scripts in `data_prepare/` directory
2. Use `data_prepare/6-csv2h5.py` to convert CSV point clouds to H5 format

For original ShapeNet data, point clouds can be downloaded [here](https://github.com/stevenygd/PointFlow).

For completion, we use ShapeNet rendering provided by [GenRe](https://github.com/xiumingzhang/GenRe-ShapeHD).
We provide script `convert_cam_params.py` to process the provided data.

For training the model on shape completion, we need camera parameters for each view
which are not directly available. To obtain these, simply run 
```bash
$ python convert_cam_params.py --dataroot DATA_DIR --mitsuba_xml_root XML_DIR
```
which will create `..._cam_params.npz` in each provided data folder for each view.

## Pretrained models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1Q7aSaTr6lqmo8qx80nIm1j28mOHAHGiM?usp=sharing).

## Training:

For NUP96 point cloud generation:
```bash
$ python train_generation.py --dataroot /path/to/nup96/data --npoints 2048 --bs 16
```

For ShapeNet (original usage):
```bash
$ python train_generation.py --category car|chair|airplane
```

Please refer to the python file for optimal training parameters.

## Testing:

```bash
$ python train_generation.py --category car|chair|airplane --model MODEL_PATH
```

## TensorBoard

Training logs and visualizations are saved to `runs/` directory. View with:
```bash
$ tensorboard --logdir runs/
```

## Results

Some generation and completion results are as follows.
<p align="center">
  <img src="assets/gen_comp.gif" width="60%"/>
</p>

Multimodal completion on a ShapeNet chair.
<p align="center">
  <img src="assets/mm_shapenet.gif" width="80%"/>
</p>


Multimodal completion on PartNet.
<p align="center">
  <img src="assets/mm_partnet.gif" width="80%"/>
</p>


Multimodal completion on two Redwood 3DScan chairs.
<p align="center">
  <img src="assets/mm_redwood.gif" width="80%"/>
</p>

## Reference

```
@inproceedings{Zhou_2021_ICCV,
    author    = {Zhou, Linqi and Du, Yilun and Wu, Jiajun},
    title     = {3D Shape Generation and Completion Through Point-Voxel Diffusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5826-5835}
}
```

## Acknowledgement

For any questions related to codes and experiment setting, please contact [Linqi Zhou](linqizhou@stanford.edu) and [Yilun Du](yilundu@mit.edu). 

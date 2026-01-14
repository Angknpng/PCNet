# PCNet
Code and dataset repository for our paper entilted "Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network" accepted at AAAI 2025.

arXiv version: https://arxiv.org/pdf/2412.14576.

***The model and results are available now. [17th, Jul, 2025]***

Thank you for your attention. 


## Dataset

[![avatar](https://github.com/Angknpng/PCNet/raw/main/Fig/dataset.png)](https://github.com/Angknpng/PCNet/blob/main/Fig/dataset.png)

<!-- The proposed UVT20K dataset link can be found here. [[baidu pan](https://pan.baidu.com/s/1LCEvXR3gKvIZOdMgZbSRVg?pwd=fete) fetch code: fete] -->

The compressed UVT20K dataset containing the annotations of saliency maps, edges, scribbles, and challenge attributes can be found here. [[baidu pan](https://pan.baidu.com/s/1CBLCup7VzU2-O2U8Aqw8oQ?pwd=v2rc) fetch code: v2rc] or [[google drive](https://drive.google.com/file/d/1u6FFnaoaJrmGyjtPD6FyI500uKDlZ6nh/view?usp=sharing)]

## Method

[![avatar](https://github.com/Angknpng/PCNet/raw/main/Fig/method.png)](https://github.com/Angknpng/PCNet/blob/main/Fig/method.png)


## Results

The predicted results of our models can be found here. [[baidu pan](https://pan.baidu.com/s/1YrpRoF6M6pt_HUOemhrofg?pwd=eekm) fetch code: eekm]

The parameters of our models can be found here. [[baidu pan](https://pan.baidu.com/s/13iUOoKOjr4PZnJ34WSrymA?pwd=gvvw) fetch code: gvvw]

The predicted results of the comparison methods can be found here. [[baidu pan](https://pan.baidu.com/s/1usGwv7SLuS7T7dKPCUwbFg?pwd=6qqn) fetch code: 6qqn]

## Usage

### Requirement

0. Download the UVT20K dataset for training and testing.
1. Download the pretrained parameters of the backbone from here. [[baidu pan](https://pan.baidu.com/s/1sBuu7Qw9n8aWRydQsDieBA?pwd=3ifw) fetch code: 3ifw]
2. Download the pretrained parameters of the IHN model from [here](https://github.com/imdumpl78/IHN).
3. Organize dataset and pretrained model directories.
4. Create directories for the experiment and parameter files.
5. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
6. Install other packages: `pip install -r requirements.txt`.
7. Set your path of all datasets in `./options.py`.

### Train

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2212 train_parallel.py
```

### Test

```
python test_produce_maps.py
```


## Citation
If you think our work is helpful, please cite:

```
@inproceedings{wang2025alignment,
  title={Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network},
  author={Wang, Kunpeng and Chen, Keke and Li, Chenglong and Tu, Zhengzheng and Luo, Bin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7780--7788},
  year={2025}
}
```


## Acknowledgement

The implement of this project is based on the following link.

- [Multi-modal homography estimator](https://github.com/imdumpl78/IHN)
- [SOD Literature Tracking](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)

## Contact

If you have any questions, please contact us (kp.wang@foxmail.com).

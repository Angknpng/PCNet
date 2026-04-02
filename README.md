# PCNet
Code and dataset repository for our AAAI 2025 paper: **"Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network"**.

📄 arXiv: https://arxiv.org/pdf/2412.14576

***The model and results are available now. [17th, Jul, 2025]***

Thank you for your attention.

---

✨ **Update (2026-01): Google Drive links added (recommended for international users).**  
- 📦 **Dataset (UVT20K compressed):** 👉 [[Google Drive](https://drive.google.com/file/d/1u6FFnaoaJrmGyjtPD6FyI500uKDlZ6nh/view?usp=sharing)]  
- 📌 **Results & checkpoints (full mirror):** 👉 [[Google Drive](https://drive.google.com/drive/folders/1C_3PdWbeG1vX6QLxwOY-KZcN960sxkvY?usp=sharing)]

---

## Dataset

[![avatar](https://github.com/Angknpng/PCNet/raw/main/Fig/dataset.png)](https://github.com/Angknpng/PCNet/blob/main/Fig/dataset.png)

The compressed **UVT20K** dataset contains annotations of **saliency maps, edges, scribbles**, and **challenge attributes**.  
Download here:
- 📦 [[Baidu Pan](https://pan.baidu.com/s/1CBLCup7VzU2-O2U8Aqw8oQ?pwd=v2rc)] (code: `v2rc`)
- 🌍 [[Google Drive](https://drive.google.com/file/d/1u6FFnaoaJrmGyjtPD6FyI500uKDlZ6nh/view?usp=sharing)] (recommended)

<!-- The proposed UVT20K dataset link can be found here. [[baidu pan](https://pan.baidu.com/s/1LCEvXR3gKvIZOdMgZbSRVg?pwd=fete) fetch code: fete] -->

---

## Method

[![avatar](https://github.com/Angknpng/PCNet/raw/main/Fig/method.png)](https://github.com/Angknpng/PCNet/blob/main/Fig/method.png)

---

## Results

✨ **Google Drive mirror (recommended for international users):**  
All released results/checkpoints (same content as the Baidu Pan links below):  
👉 [[Google Drive](https://drive.google.com/drive/folders/1C_3PdWbeG1vX6QLxwOY-KZcN960sxkvY?usp=sharing)]

- 📌 **Predicted results (ours):** [[Baidu Pan](https://pan.baidu.com/s/1YrpRoF6M6pt_HUOemhrofg?pwd=eekm)] (code: `eekm`)
- 🧩 **Model checkpoints:** [[Baidu Pan](https://pan.baidu.com/s/13iUOoKOjr4PZnJ34WSrymA?pwd=gvvw)] (code: `gvvw`)
- 📊 **Predicted results (compared methods):** [[Baidu Pan](https://pan.baidu.com/s/1usGwv7SLuS7T7dKPCUwbFg?pwd=6qqn)] (code: `6qqn`)

---

## Usage

### Requirements

0. 📦 Download **UVT20K** for training and testing (see **Dataset** section above).
1. 🧠 Download the pretrained backbone parameters:  
   - 📦 [[Baidu Pan](https://pan.baidu.com/s/1sBuu7Qw9n8aWRydQsDieBA?pwd=3ifw)] (code: `3ifw`)
2. 🧩 Download the pretrained parameters of **IHN** from: [[IHN](https://github.com/imdumpl78/IHN)].
3. 📁 Organize dataset and pretrained model directories.
4. 🗂️ Create directories for experiments and checkpoints.
5. 🧪 Install PyTorch via `conda`: `torch==1.12.0`, `torchvision==0.13.0`.
6. 📦 Install other packages: `pip install -r requirements.txt`.
7. 🔧 Set dataset paths in `./options.py`.

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

This project is based on the following resources:

- [Multi-modal homography estimator](https://github.com/imdumpl78/IHN)
- [SOD Literature Tracking](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)
- [Matlab-based Evaluation](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox)

## Contact

📮 For questions or feedback, feel free to email: kp.wang@foxmail.com

## GLFA
This is the official PyTorch implementation of the paper: ["Enhancing Cross-Subject fMRI-to-Video Decoding with Global-Local Functional Alignment"](https://link.springer.com/chapter/10.1007/978-3-031-73010-8_21)

### Abstract

<img src=".\assets\diagram.png" alt="diagram" style="zoom: 15%;" />

Advancements in brain imaging enable the decoding of thoughts and intentions from neural activities. However, the fMRI-to-video decoding of brain signals across multiple subjects encounters challenges arising from structural and coding disparities among individual brains, further compounded by the scarcity of paired fMRI-stimulus data. Addressing this issue, this paper introduces the fMRI Global-Local Functional Alignment (GLFA) projection, a novel approach that aligns fMRI frames from diverse subjects into a unified brain space, thereby enhancing cross-subject decoding. Additionally, we present a meticulously curated fMRI-video paired dataset comprising a total of 75k fMRI-stimulus paired samples from 8 individuals. This dataset is approximately 4.5 times larger than the previous benchmark dataset. Building on this, we augment a transformer-based fMRI encoder with a diffusion video generator, delving into the realm of cross-subject fMRI-based video reconstruction. This innovative methodology faithfully captures semantic information from diverse brain signals, resulting in the generation of vivid videos and achieving an impressive average accuracy of 84.7% in cross-subject semantic classification tasks.

### Installation

Download this repository and create the environment:

```
git clone https://github.com/chongjg/GLFA-fmri-video.git
cd GLFA-fmri-video
conda create -n GLFA python=3.10
conda activate GLFA
pip install -r requirements.txt
```

### Getting Start

##### 1. Download the Dataset

You can download our fMRI-Video dataset by this link: https://huggingface.co/datasets/Fudan-fMRI/fMRI-Video.

##### 2. Download the Zeroscope

Download the [zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main) and place it in the following directory:

```
./models/zeroscope
```

##### 3.Download the Encoder

Download the [encoder](https://drive.google.com/file/d/13und4QhjwJcD5B2GhmMRcheAis2-4fm9/view?usp=sharing) and place it in the following directory:

```
./models/fmri_encoder
```

##### 4. Stage 1: Contrastive Learning

```
python fmri_text_align.py
```

##### 5. Stage 2: Co-training

```
python fmri_video.py
```

##### 6. inference & evaluation

Checkpoint of pipeline can be downloaded in [fmri-video-sub12-funcalign.zip](https://drive.google.com/file/d/1xaTANPmTx509EBny69RTaazX_LAxM1w1/view?usp=sharing) and generated samples can be downloaded in [samples.zip](https://drive.google.com/file/d/17YFthQ7RqO5c7ENEAmTiwLfmEJ8W3zzV/view?usp=sharing).

```
python inference.py
python run_metrics.py ./path/to/samples/
```

### Results

<img src=".\assets\within-ourdataset.jpeg" alt="within-ourdataset" style="zoom:15%;" />

<img src=".\assets\within-Wen.jpeg" alt="within-Wen" style="zoom:20%;" />

### Citation

If you find our work useful to your research, please consider citing:

```
@inproceedings{li2024enhancing,
  title={Enhancing cross-subject fmri-to-video decoding with global-local functional alignment},
  author={Li, Chong and Qian, Xuelin and Wang, Yun and Huo, Jingyang and Xue, Xiangyang and Fu, Yanwei and Feng, Jianfeng},
  booktitle={European Conference on Computer Vision},
  pages={353--369},
  year={2024},
  organization={Springer}
}
```

### Acknowledgement

This project is built on [animate-anything](https://github.com/alibaba/animate-anything). Evaluation code is adapted from [MindVideo](https://github.com/jqin4749/MindVideo). Thanks for their excellent work.


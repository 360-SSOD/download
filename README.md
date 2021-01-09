
### Requirements

- Linux
- Python 3.5+ ([Say goodbye to Python2](https://python3statement.org/))
- PyTorch 1.0+ or PyTorch-nightly
- CUDA 9.0+
- NCCL 2+
- GCC 4.9+
- [mmcv](https://github.com/open-mmlab/mmcv)

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04 and CentOS 7.2
- CUDA: 9.0/9.2/10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC: 4.9/5.3/5.4/7.3

### Install mmdetection

1. Create a conda virtual environment and activate it. Then install Cython.

2. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/).

3. Clone the mmdetection repository.

```shell
git clone https://github.com/360-SSOD/download.git
cd mmdetection
```

4. RPN models are available in the [Model](https://github.com/NANSHANB/GARP/blob/master/MODEL_ZOO.md)

### Datasets and Models:

360-SSOD : [Drive Link](https://drive.google.com/file/d/1jTSAnok9nUbEO8FP9V2C70GztNdkWWGw/view?usp=sharing)

Models : [Drive Link](https://drive.google.com/file/d/1NSGToSjVXuteDxHV7ROj_pfpT8B_T0WO/view?usp=sharing) 


### Usage

1. Please download our pretrained model at [Drive Link](https://drive.google.com/file/d/1NSGToSjVXuteDxHV7ROj_pfpT8B_T0WO/view?usp=sharing).   
   Put this model in `./model/` and `./reimg/`.

2. Run:   `test.py`
    

### Evaluation Method

Saliency Evaluation Method : The code  can be found in [Ming-Ming Cheng](http://mmcheng.net) and [Deng-Ping Fan](http://dpfan.net/).

### Citation:

```
@article{ma2020stage,
  title={Stage-wise salient object detection in 360° omnidirectional image via object-level semantical saliency ranking},
  author={Ma, Guangxiao and Li, Shuai and Chen, Chenglizhao and Hao, Aimin and Qin, Hong},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={26},
  number={12},
  pages={3535--3545},
  year={2020},
  publisher={IEEE}
}
```

### Contacts
mgx@jbeta.net


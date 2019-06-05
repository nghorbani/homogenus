# Human Image Gender Classifier


This is the official repository of the Humun Gender Classifier (homogenus) used in the paper Expressive Body Capture:
3D Hands, Face, and Body from a Single Image. The codebase consists of the inference code, i.e. given a full human body image, and the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)  output json files using this code one can augment the joint detections of the openpose with the gender of the detected human. The output classes would be either male,female, or neutral for undetected genders. For further details on the method please refer to the following publication,

```
Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
G. Pavlakos*, V. Choutas*, N. Ghorbani, T. Bolkart, A. A. A. Osman, D. Tzionas and M. J. Black 
Computer Vision and Pattern Recognition (CVPR) 2019, Long Beach, CA
```

A pdf preprint is also available on the [project page](https://smpl-x.is.tue.mpg.de/).


## Installation

The code uses Python 2.7 and it is tested on Tensorflow gpu version 1.12.0, with CUDA-9.0 and cuDNN-7.3.

### Setup homogenus Virtual Environment

```
virtualenv --no-site-packages <your_home_dir>/.virtualenvs/homogenus
source <your_home_dir>/.virtualenvs/homogenus/bin/activate
```
### Clone the project and install requirements

```
git clone https://github.com/nghorbani/homogenus.git
cd homogenus
pip install -r requirements.txt
pip install opendr==0.77
mkdir model
```

## Download models

* Download pretrained homogenus weights from the [project website](https://smpl-x.is.tue.mpg.de), downloads page. Copy this inside the **model** folder

#### Output predicted gender labels

If you want the output mesh then run the following command
```
python -m demo --img_path 
```
It will either overwrite the original  openpose json files or create new ones in your specified path.



## License

Free for non-commercial and scientific research purposes. By using this code, you acknowledge that you have read the license terms (https://smpl-x.is.tue.mpg.de/license), understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not use the code. For commercial use please check the website (https://smpl-x.is.tue.mpg.de/license).

## Referencing SMPL-X

Please cite the following paper if you use the code directly or indirectly in your research/projects.
```
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```

## Contact

If you have any questions you can contact us at smplx@tuebingen.mpg.de

## Acknowledgement

* We thank Benjamin Pellkofer and Jonathan Williams for helping with our [smpl-x project website](https://smpl-x.is.tue.mpg.de).
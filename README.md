## EvidenceCap: Trusworthy medical image segmentation
Conference version of 
[TBraTS: Trusted Brain Tumor Segmentation](https://arxiv.org/abs/2206.09309)

Journal version of
[EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap](https://arxiv.org/abs/2301.00349)

## Requirements
Some important required packages include:  
Pytorch version >=0.4.1.  
Visdom  
Python == 3.7  
Some basic python packages such as Numpy.  

## Introduction
Medical image segmentation (MIS) is essential for supporting disease diagnosis and treatment effect assessment. Despite considerable advances in artificial intelligence (AI) for MIS, clinicians remain skeptical of its utility, maintaining low confidence in such black box systems, with this problem being exacerbated by low generalization for out-of-distribution (OOD) data. To move towards effective clinical utilization, we propose a foundation model named EvidenceCap, which makes the box transparent in a quantifiable way by uncertainty estimation. EvidenceCap not only makes AI visible in regions of uncertainty and OOD data, but also enhances the reliability, robustness, and computational efficiency of MIS. Uncertainty is modeled explicitly through subjective logic theory to gather strong evidence from features. We show the effectiveness of EvidenceCap in three segmentation datasets and apply it to the clinic. Our work sheds light on clinical safe applications and explainable AI, and can contribute towards trustworthiness in the medical domain.

<div align=center><img width="900" height="300" alt="Our TBraTS framework" src="https://github.com/Cocofeat/UMIS/blob/main/image/Moti-TMIS.png"/></div>
<!--
<div align=center><img width="900" height="400" alt="Our TBraTS framework" src="https://github.com/Cocofeat/UMIS/blob/main/image/NC_F1.png"/></div>
-->

## Original Data Acquisition
- The ISIC2018 dataset could be acquired from [here](https://challenge2018.isic-archive.com/).
- The LiTS2017 dataset could be acquired from [here](https://competitions.codalab.org/competitions/17094).
- The BraTS2019 dataset could be acquired from [here](https://ipp.cbica.upenn.edu/).
- The Johns Hopkins OCT dataset could be acquired from [here](https://iacl.ece.jhu.edu/index.php?title=Main_Page).
- The Duke OCT dataset with DME dataset could be acquired from [here](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm).
- The DRIVE dataset could be acquired from [here](https://drive.grand-challenge.org/DRIVE/).
- The FIVES dataset could be acquired from [here](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1).

## Data Preprocess or download Noised data directly 
- ISIC2018 
After downloading the dataset from [here](https://challenge2018.isic-archive.com/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.
Follow the `python3 data/preprocessISIC.py ` which is referenced from the [CA-Net](https://github.com/HiLab-git/CA-Net/blob/master/isic_preprocess.py)
- create noise data 

Follow the Follow the `python3 data/isic_condition_list.py `

- LiTS2017 
After downloading the dataset from [here](https://competitions.codalab.org/competitions/17094), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

Follow the `python3 data/preprocessLiver.py ` which is referenced from the [H-DenseU](https://github.com/xmengli/H-DenseUNet/blob/master/preprocessing.py)

Follow the Follow the `python3 data/LiTS_condition_list.py `

- BraTS2019
- After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and realize date normalization.

Follow the `python3 data/preprocessBraTS.py ` which is referenced from the [TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/data/preprocess.py)

- create noise data 

Follow the Follow the `python3 data/brats_condition_list.py `


## Noised Data Acquisition
### Task1: Skin lession segmentation
- The ISIC2018 dataset with Gaussian noise and random pixel mask could be acquired from google drive [here](https://challenge2018.isic-archive.com/).
### Task2: Liver segmentation
- The LiTS2017 dataset with Gaussian noise, Gaussian blur and random pixel mask could be acquired from google drive [here](https://competitions.codalab.org/competitions/17094).
### Task3: Brain tumor segmentation
- The BraTS2019 dataset with Gaussian noise, Gaussian blur and random pixel mask  could be acquired from  google drive [here](https://ipp.cbica.upenn.edu/).



## Training & Testing 
Run the `python3 pretrainUMIS.py ` : your own backbone with our framework(U/V/AU/TransBTS)

Run the `python3 trainUMIS.py ` : the backbone without our framework

##  :fire: NEWS :fire:
* [01/05] We have released the codes. 
* [01/01] We will release the code as soon as possible. 
* Happy New Year!
* [09/17] We will release the code as soon as possible. 

If you find our work is helpful for your research, please consider to cite:  
```
@misc{Coco2022EvidenceCap,
  author    = {Zou, Ke and Yuan, Xuedong and Shen, Xiaojing and Wang, Meng and Rick, Siow Mong, Goh and Liu, Yong and Fu, Huazhu},
  title     = {EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap},
  year      = {2023},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2301.00349},

}
}
```

```
@InProceedings{Coco2022TBraTS,
  author    = {Zou, Ke and Yuan, Xuedong and Shen, Xiaojing and Wang, Meng and Fu, Huazhu},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  title     = {TBraTS: Trusted Brain Tumor Segmentation},
  year      = {2022},
  address   = {Cham},
  pages     = {503--513},
  publisher = {Springer Nature Switzerland},
}
}
```

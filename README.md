# EvidenceCap: Trusworthy medical image segmentation
- Conference version of 
[[Paper]](https://arxiv.org/abs/2206.09309)  [[code]](https://github.com/Cocofeat/TBraTS)

- Journal version of
[[Paper]](https://arxiv.org/abs/2301.00349)  [[code]](https://github.com/Cocofeat/UMIS)

## 1. Requirements
    Some important required packages include:    
    - Python == 3.6  
    - pytorch 1.6.0
    - torchvision 0.7.0
    Some basic python packages such as Numpy...  

## 2. Overview

### 2.1 Introduction
Medical image segmentation (MIS) is essential for supporting disease diagnosis and treatment effect assessment. Despite considerable advances in artificial intelligence (AI) for MIS, clinicians remain skeptical of its utility, maintaining low confidence in such black box systems, with this problem being exacerbated by low generalization for out-of-distribution (OOD) data. To move towards effective clinical utilization, we propose a foundation model named EvidenceCap, which makes the box transparent in a quantifiable way by uncertainty estimation. EvidenceCap not only makes AI visible in regions of uncertainty and OOD data, but also enhances the reliability, robustness, and computational efficiency of MIS. Uncertainty is modeled explicitly through subjective logic theory to gather strong evidence from features. We show the effectiveness of EvidenceCap in three segmentation datasets and apply it to the clinic. Our work sheds light on clinical safe applications and explainable AI, and can contribute towards trustworthiness in the medical domain.

<div align=center><img width="900" height="300" alt="Our TBraTS framework" src="https://github.com/Cocofeat/UMIS/blob/main/image/Moti-TMIS.png"/></div>


### 2.2 Framework Overview
EvidenceCap is a trustworthy medical image segmentation framework based on evidential deep learning, which provides robust segmentation performance and reliable uncertainty quantification for diagnostic support. A pipeline of EvidenceCap and its results in undertaking trustworthy medical image segmentation tasks are shown in Fig. 1 b and c. In the training phase (Fig. 1 b), EvidenceCap can be applied to any task in numerous medical domains. Its trained model visually generates auxiliary diagnostic results, including robust target segmentation results and reliable uncertainty estimation. In the testing phase, in order to verify the effectiveness of the method, EvidenceCap was tested for confidence, robustness, and computational efficiency on different segmentation tasks.

<div align=center><img width="900" height="400" alt="Our EvidenceCap framework" src="https://github.com/Cocofeat/UMIS/blob/main/image/NC_F1.png"/></div>


### 2.3 Qualitative Results

<div align=center><img width="900" height="400" alt="Qualitative Results on BraTS2019 dataset" src="https://github.com/Cocofeat/UMIS/blob/main/image/brats_fA2.png"/></div>

## 3. Proposed Baseline

### 3.1 Original Data Acquisition
- The ISIC2018 dataset could be acquired from [here](https://challenge2018.isic-archive.com/).
- The LiTS2017 dataset could be acquired from [here](https://competitions.codalab.org/competitions/17094).
- The BraTS2019 dataset could be acquired from [here](https://ipp.cbica.upenn.edu/).
- The Johns Hopkins OCT dataset could be acquired from [here](https://iacl.ece.jhu.edu/index.php?title=Main_Page).
- The Duke OCT dataset with DME dataset could be acquired from [here](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm).
- The DRIVE dataset could be acquired from [here](https://drive.grand-challenge.org/DRIVE/).
- The FIVES dataset could be acquired from [here](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1).

### 3.2 Data Preprocess or download Abnormal data directly 

Abnormal data: Gaussian noise, Gaussian blur and random pixel mask

- Task1: ISIC2018 

    + Preprocess
    
        After downloading the dataset from [here](https://challenge2018.isic-archive.com/), data preprocessing is needed. 
        
        Follow the `python3 data/preprocessISIC.py ` which is referenced from the [CA-Net](https://github.com/HiLab-git/CA-Net/blob/master/isic_preprocess.py)
        
    + Create noise data (Gaussian noise and Random mask) 
    
       Follow the `python3 data/isic_condition_list.py ` which is preprocessed to create noised data for ISIC.

- Task2: LiTS2017 

    + Preprocess
    
        After downloading the dataset from [here](https://competitions.codalab.org/competitions/17094), data preprocessing is needed. 
        
        Follow the `python3 data/preprocessLiver.py ` which is referenced from the [H-DenseU](https://github.com/xmengli/H-DenseUNet/blob/master/preprocessing.py)

    + Create the abnormal data (Gaussian noise, blur and Random mask)

        Follow the `python3 data/LiTS_condition_list.py `  which is preprocessed to create noised data for Liver.

- Task3: BraTS2019

    + Preprocess
    
        After downloading the dataset from [here](https://ipp.cbica.upenn.edu/), data preprocessing is needed which is to convert the .nii files as .pkl files and             realize date normalization. 
        
        Follow the `python3 data/preprocessBraTS.py ` which is referenced from the [TransBTS](https://github.com/Wenxuan-1119/TransBTS/blob/main/data/preprocess.py)

    + Create the noised data (Gaussian noise, blur and Random mask) 
    
        Follow the `python3 data/brats_condition_list.py `


### 3.3 Abnormal Data Acquisition
- Task1: Skin lession segmentation
    + The ISIC2018 dataset with Gaussian noise and random pixel mask could be acquired from google drive [here](https://drive.google.com/file/d/1BioeT7NR6bZ2xSqDwVd7BYrZd2ZjLD_-/view?usp=share_link).
   
- Task2: Liver segmentation
    + The LiTS2017 dataset with Gaussian noise, Gaussian blur and random pixel mask could be acquired from google drive [here](https://drive.google.com/file/d/1zHc91xKvZNv-liSHdhKy9efn9RV4l08Q/view?usp=share_link).
    
- Task3: Brain tumor segmentation
    + The BraTS2019 dataset with Gaussian noise, Gaussian blur and random pixel mask  could be acquired from  google drive [here](https://drive.google.com/file/d/1hWFKidJaLEXSYdn6N7QAPAkLtVbh49o6/view?usp=share_link).

<!--
-->

### 3.4 Training & Testing 
- Training Configuration:
    + Run the `python3 pretrainUMIS.py ` and change the `mode = train` : your own backbone with our framework(U/V/AU/TransBTS)
        Just enjoy it!
        
    + Run the `python3 trainUMIS.py ` and change the `mode = train`  : the backbone without our framework
        Just enjoy it!
        
- Testing Configuration:
    + Run the `python3 pretrainUMIS.py ` and change the `mode = test`, `OOD_Condition`, `OOD_Level`, `python3 model_name`  : your own backbone with our framework(U/V/AU/TransBTS)
        Just enjoy it!
        
    + Run the `python3 trainUMIS.py `  and change the `mode = test`, `OOD_Condition`, `OOD_Level`, `python3 model_name`   : the backbone without our framework
        Just enjoy it!
        
### 3.5 Evaluating & Output uncertainty map
- Evaluating:
    + Refer to the `python3 predict.py`
    
- Output uncertainty map
    + Refer to the `python3 predict.py` and the function of  'test' in `python3 trainUMIS.py` 
    
##  :fire: NEWS :fire:
* [01/05] We have released the codes. 
* [01/01] We will release the code as soon as possible. 
* Happy New Year!

If you find our work is helpful for your research, please consider to cite:  
```
@misc{Coco2022EvidenceCap,
  author    = {Zou, Ke and Yuan, Xuedong and Shen, Xiaojing and Wang, Meng and Rick, Siow Mong, Goh and Liu, Yong and Fu, Huazhu},
  title     = {EvidenceCap: Towards trustworthy medical image segmentation via evidential identity cap},
  year      = {2023},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2301.00349},
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
```

## Acknowledgement
Part of the code is revised from [CA-Net](https://github.com/HiLab-git/CA-Net), [H-DenseU](https://github.com/xmengli/H-DenseUNet), [TransBTS](https://github.com/Wenxuan-1119/TransBTS), [TMC](https://github.com/hanmenghan/TMC)

## Contact
* If you have any problems about our work, please contact [me](kezou8@gmail.com) 
* Project Link: [TBraTS](https://github.com/Cocofeat/TBraTS/); [EvidenceCap](https://github.com/Cocofeat/UMIS/)

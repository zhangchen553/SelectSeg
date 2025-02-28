# SelectSeg: Uncertainty-based selective training and prediction for accurate crack segmentation under limited data and noisy annotations
This repository contains the codes for selective crack image segmentation. The methodology hereby implemented was presented in the paper:

![Framework](/images/framework.png "Framework")

## 1. Envrionment
`conda create -n sel_seg python=3.10`

`conda activate sel_seg`

`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

`pip install -r requirements.txt`

## 2. Repository directory

The repository directory should look as:

```
SelectSeg
└───framework  # The proposed method
└───DATASET
    └───CRACK500  # needs to download
└───other_denoise  # Other denoise methods for comparison
└───other_SSL  # Ohter semi-supervised learning methods for comparison
└───utils
└───Results_CRACK500
```
## 3. Dataset
Download the CRACK500 dataset from [here](https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001).   
Download the noisy annotations for CRACK500 from [here](https://data.mendeley.com/datasets/wddt4gbttd/1).  
The ratio of the noisy annotations can be adjusted.

The dataset directory should look as:  
(Separate image and mask into different folder. Both image and mask should have the same name with .png format)
```
└───CRACK500
    └───train_crop_image
    └───train_crop_mask_10pct 
    └───train_crop_mask_xxpct  # The xx percent annotations are used as noisy annotations
    └───val_crop_image
    └───val_crop_mask
    └───test_crop_image
    └───test_crop_mask
```

## 4. Model
The model is based on the Segmentation Model Pytorch [repository](https://github.com/qubvel-org/segmentation_models.pytorch), 
and will be automatically downloaded when running the code.  
The default model is a Feature Pyramid Network (FPN) with a segformer (MiT) backbone can be modified in `a_config.py`.  

The trained models are named according to training configuration.   
For example, `100pct_CRACK500_FPN_mit_b2_Augbase_LR2E-04_LOSSDice_SEED1.pth`

## 5. Selective Training
Each of the main.py has it corresponding config.py  
E.g: e_main.py —> a_config.py
### 5.1 First stage:
To train the models with the provided dataset, run the following scripts:  
`cd framework`  
`conda activate sel_seg`  
`python e_main_script.py`   
Four models trained with different random seeds will be saved in the `Results_CRACK500/full_supervise_xxpct` directory.
(xx is the corresponding ratio of noisy annotation) This stage prepares the models for the next stage.

### 5.2 Second stage: 
#### Step 1: Calculate the uncertainty score for each image-mask pair. 
`python c_ann_score.py --c_ratio=xx`  # xx is the ratio of noisy annotation used in the training model.  
This script will output a `c_CRACK500_train_f1_vs_f1_var_mses.txt` in `Results_CRACK500/full_supervise_xxpct`, 
which is used to determine the uncertainty score for each image-mask pair.

#### Step 2: Rank the images based on the uncertainty score. 
`python c_data_ranking.py --c_ratio=xx --portion 90 80 70 60 50 40 30 20 10`  
According to the ranking score, the script will output different ratio of image list 
in `Results_CRACK500/full_supervise_xxpct/data_split_txt`.  
The portion can be adjusted according to the requirement for the grid search.  

For example:  
`sup_CRACK500_train_90pct.txt` is the list of 90% high ranking images.  
`unsup_CRACK500_train_10pct.txt` is the list of 10% low ranking images.  

The txt files in the example are ranked with our proposed method.  
The other txt files ended with other name (like `xxpct_f1_f1_var`) after `xxpct` are 
the ranking list under different score metric for comparison.

### 5.3 Third stage:
#### Step 1: Train the models with the high ranking images.
New models are trained with the high ranking images from the previous stage.  

Run the following scripts: `python e_main_partial_fully.py --c_ratio=50 --portion 60`  

This script needs to be customized and can help to decide the rankings' threshold.  
The models will be saved in the `Results_CRACK500/partial_supervise_xxpct` directory.

#### Step 2: Apply Semi-Supervised Learning (SSL) to train the model with best ranking threshold.
Cross Pseudo Supervision (CPS) is used to train the model with all the images.

Run the following scripts:
`python e_cps_main.py --c_ratio=50 --portion 60`  
If you would like to apply selective evaluation, run the script: `python e_cps_main_script.py`  
The model will be saved in the `Results_CRACK500/cps_50pct` directory.

## 6. Evaluation:
### 6.1 Regular Evaluation:
`python f_evaluation.py --c_ratio=xx`

### 6.2 Selective Evaluation - Fourth stage (optional):
`python c_ann_score.py --c_ratio=xx --flag=test_`  
`python g_sel_eval.py --c_ratio=xx --flag=test_`  
The script will output the evaluation results in the `Results_CRACK500/cps_xx_pct` directory.

## 7. Comparison with the other methods
### 7.1 Semi-supervised learning methods
Our implementation of the semi-supervised learning methods can be found in the `other_SSL` directory.
### 7.2 Denoising methods
Our implementation of the denoising methods can be found in the `other_denoise` directory.  
`SCE`, `T-Loss`, `LS`, and `Soft Label` share the same training script:  
For loss `SCE` and `T-Loss`, `python e_main.py --loss=xxx`.  
For `LS`, `python e_main.py --smooth=xxx`.  
For `Soft Label`, it needs to generate the soft label first, and configure the `a_config.py` to use the soft label.

The training script of `ADELE` and `SFL` can be found in the `other_denoise` directory.  

![Predictions](/images/ours_baseline.png "Comparison")

## 8. Training with your data and model.
Follow the structure of the data/ folder and place your data accordingly.  
Please refer to the Segmentation Model Pytorch [repository](https://github.com/qubvel-org/segmentation_models.pytorch) for more models.

## 9. Citation
If you find this repository useful, please cite the following paper:
https://doi.org/10.1016/j.ress.2025.110909
```
@article{

}
```

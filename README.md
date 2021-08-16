
Xiangru:
Some changes or places that animal might be different with human:

1. datasets/dfaust.py: 
Remove split and test_exp in def __init__(), so the data path would be processed/training.pt and processed/testing.pt 

2. main.py 
L30: model channel is [16, 16, 16, 32] for animal, but need [32, 32, 32, 64] for human.

L129: ds_factors = [1, 1, 1, 1] for animal, need ds_factors = [2, 2, 2, 2] for human. 

L179: torch.nn.init.normal_(test_lat_vecs.weight.data, 0.0, 0.2) init test vector with mean 0, variance 0.2. but for human might be 0, 1.0

L48: --arap_eig_k: only care about last k eigvals of the hessian matrix in ARAP. This feature is added after iccv submission. If we set k=0, it’s equal to the old experiment. 
This is for shape pose disentanglement. We only care about small eigenvalues in tracing because they are pose parameters. For animal, we have 96 dimensional latent vector. We set k=30, so we only care about eigvals[30: -1].  

3. Utils/train_eval.py
L80: the Jacobian function is for animal where there’s no stochastic sampling. You can change to your own function. 



# ARAPReg
Code for ICCV 2021 paper: [ARAPReg: An As-Rigid-As Possible Regularization Loss for LearningDeformable Shape Generators.](arxiv_link).

## Data Preparation
We provide data paths and loaders for 3 datasets: [DFAUST](https://dfaust.is.tue.mpg.de/), [SMAL](https://smal.is.tue.mpg.de/) and Bone dataset. 

### DFAUST
We use 4264 test shapes and 32933 training shapes from DFaust dataset.
You can download the dataset [here](https://drive.google.com/file/d/1BaACAdJO0uoH5P084Gw11a_j3nKVSUjn/view?usp=sharing).
Please place `dfaust.zip` in `data/DFaust/raw/`.

### SMAL
We use 400 shapes from the family 0 in SMAL dataset. We generate shapes by the SMAL demo where the mean and the variance of the pose vectors are set to 0 and 0.2. We split them to 300 training and 100 testing samples. 

You can download the generated dataset [here](https://drive.google.com/file/d/1L3n6i097bgZtNYAmnGM9NwOWBNd4c1Fr/view?usp=sharing).
After downloading, please move the downloaded `smal.zip` to `./data/SMAL/raw`.

### Bone
We created a conventional bone dataset with 4 categories: tibia, pelvis, scapula and femur. Each category has about 50 shapes. We split them to 40 training and 10 testing samples. 
You can download the dataset [here](https://drive.google.com/file/d/1Naq1F6V-Oxw4AQZJeaCKfRrOCQneF0gT/view?usp=sharing).
After downloading, please move `bone.zip` to `./data` then extract it. 


## Testing
### Pretrained checkpoints
You can find pre-trained models and training logs in the following paths:

**DFAUST**: [TODO]

**SMAL**: [smal_ckpt.zip](https://drive.google.com/file/d/1IIAW5SmylMHsFpU-croeu-uNPdKP_fnL/view?usp=sharing).  Move it to `./work_dir/SMAL/out`, then extract it. 

**Bone**: [bone_ckpt.zip](https://drive.google.com/file/d/1pKiLV2V0DD6_izzYA6r1yNPSouA2OVW8/view?usp=sharing). Move it to `./work_dir`, then extract it. It contains checkpoints for 4 bone categories. 

### Run testing 
After putting pre-trained checkpoints to their corresponding paths, you can run the following scripts to optimize latent vectors for shape reconstruction. Note that our model has the auto-decoder architecture, so there's still a latent vector training stage for testing shapes. 

**DFAUST**:
```
bash test_dfaust.sh
```
**SMAL**:
```
bash test_smal.sh
```
**Bone**:
```
bash test_smal.sh
```
Note that for bone dataset, we train and test 4 categories seperately. Currently there's `tibia` in the training and testing script. You can replace it with `femur`, `pelvis` or `scapula` to get results for other 3 categories. 


## Model training 
To retrain our model, run the following scripts after downloading and extracting datasets. 
**DFAUST**:
```
bash train_dfaust.sh
```
**SMAL**:
```
bash train_smal.sh
```
**Bone**:
```
bash train_bone.sh
```


## Train on a new dataset
Data preprocessing and loading scripts are in `./datasets`.
To train on a new dataset, please write data loading file similar to `./datasets/dfaust.py`. Then add the dataset to `./datasets/meshdata.py` and `main.py`. Finally you can write a similar training script like `train_dfaust.sh`. 




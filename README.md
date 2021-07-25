# ARAPReg

Some changes or places that animal might be different with human:

1. datasets/dfaust.py: 
Remove split and test_exp in def __init__(), so the data path would be processed/training.pt and processed/testing.pt 


2. datasets/meshdata.py
L79:  mean and std are 0 and 1 for animal. Not sure if human need normalization.


3. main.py 
L30: model channel is [16, 16, 16, 32] for animal, but need [32, 32, 32, 64] for human.

L129: ds_factors = [1, 1, 1, 1] for animal, need ds_factors = [2, 2, 2, 2] for human. 

L179: torch.nn.init.normal_(test_lat_vecs.weight.data, 0.0, 0.2) init test vector with mean 0, variance 0.2. but for human might be 0, 1.0

L48: --arap_eig_k: only care about last k eigvals of the hessian matrix in ARAP. This feature is added after iccv submission. If we set k=0, it’s equal to the old experiment. 
This is for shape pose disentanglement. We only care about small eigenvalues in tracing because they are pose parameters. For animal, we have 96 dimensional latent vector. We set k=30, so we only care about eigvals[30: -1].  

4. Utils/train_eval.py
L80: the Jacobian function is for animal where there’s no stochastic sampling. You can change to your own function. 

# Importance Weighted Auto-Encoders Pytorch

This Code to reproduce the experiments in the Importance Weighted Auto-Encoders(IWAE) paper(2016) by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov.The implementation was tested on the MNIST dataset to replicate the result in the above paper. You can train and test VAE and IWAE with 1 or 2 stochastic layers in different configurations of K and M in this repo. 

# Prerequisites for running the code
## Dataset
Download the required dataset by running the following command. 
```
python download_MNIST.py
```
## Python packages
pytorch==1.1.0 numpy==1.14.2

# Running the experiments
This code allows you to train, evaluate and compare VAE and IWAE architectures on the mnist dataset. To train and test the model, run the following commands.
## Trainning Original VAE
```
python main_train.py  --model VAE --num_stochastic_layers 1 --num_m 1 --num_k 1
```
## Training IWAE with 2 stochastic layers
```
python main_train.py  --model IWAE --num_stochastic_layers 2 --num_m 1 --num_k 5
```
## Testing Original VAE
```
python main_test.py  --model VAE --num_stochastic_layers 1 --num_m 1 --num_k 1 --epoch 4999
```
## Testing IWAE with 2 stochastic layers
```
python main_test.py  --model IWAE --num_stochastic_layers 2 --num_m 1 --num_k 5 --epoch 4999
```
## Testing IWAE with 2 stochastic layers on log likelihood 
```
python main_test_k.py  --model IWAE --num_stochastic_layers 2 --num_m 1 --num_k 5 --num_k_test 5000 --epoch 4999
```
See [the training file](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_train.py) and [the test file](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_test.py) for more options.

### Experiment results of this repo on binarized MNIST dataset


|   Method   | NLL (This repo) | NLL ([IWAE paper](https://arxiv.org/abs/1509.00519)) | NLL ([MIWAE paper](https://arxiv.org/abs/1802.04537))|
| -----------------| --------------- | --------------- | --------|
|VAE or IWAE<sub>(M=K=1)</sub>| 86.28| 86.76| -|
|MIWAE<sub>(1,64)</sub>| | |84.52|
|MIWAE<sub>(4,16)</sub>| | |84.56 |
|MIWAE<sub>(8,8)</sub>| | |84.97 |
|MIWAE<sub>(16,4)</sub>| | | -|
|MIWAE<sub>(64,1)</sub>| |  |86.21|

|   Method   | IWAE<sub>MK</sub> loss (This repo) | IWAE<sub>MK</sub> loss ([MIWAE paper](https://arxiv.org/abs/1802.04537))|
| -----------------| ---------------| --------|
|VAE or IWAE<sub>M=K=1</sub>| 90.32| - |
|MIWAE<sub>(1,64)</sub>| |86.11|
|MIWAE<sub>(4,16)</sub>| |85.60 |
|MIWAE<sub>(8,8)</sub>| | 85.69 |
|MIWAE<sub>(16,4)</sub>| |  -|
|MIWAE<sub>(64,1)</sub>| |  86.69|

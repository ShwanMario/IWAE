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
python main_train.py  --model VAE --num_stochastic_layers 1 --num_m 5 --num_k 1
```
## Training IWAE with 2 stochastic layers
```
python main_train.py  --model IWAE --num_stochastic_layers 2 --num_m 5 --num_k 5
```
## Testing Original VAE
```
python main_train.py  --model VAE --num_stochastic_layers 1 --num_m 5 --num_k 1 --epoch 4999
```
## Testing IWAE with 2 stochastic layers
```
python main_train.py  --modelIWAE --num_stochastic_layers 2 --num_m 5 --num_k 1 --epoch 4999
```
See [the training file](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_train.py) and [the test file](https://github.com/ShwanMario/IWAE/blob/master/Importance_Weighted_Autoencoders-master/MNIST/script/main_test.py) for more options.

###Experiment results of this repo on binarized MNIST dataset
|   Method   | IWAE<sub>MK</sub> loss (This repo) | IWAE<sub>MK</sub> loss ([MIWAE paper](https://arxiv.org/abs/1802.04537))|
| -----------------| ---------------| --------|
|VAE or IWAE<sub>M=K=1</sub>| 90.32| - |
|IWAE<sub>(1,64)</sub>| |86.11|
|IWAE<sub>(4,16)</sub>| |85.60 |
|IWAE<sub>(8,8)</sub>| | 85.69 |
|IWAE<sub>(16,4)</sub>| |  -|
|IWAE<sub>(64,1)</sub>| |  86.69|



|   Method   | NLL (This repo) | NLL ([IWAE paper](https://arxiv.org/abs/1509.00519)) | NLL ([MIWAE paper](https://arxiv.org/abs/1802.04537))|
| -----------------| --------------- | --------------- | --------|
|VAE or IWAE<sub>(M=K=1)</sub>| 86.28| 86.76| -|
|IWAE<sub>(1,64)</sub>| | |84.52|
|IWAE<sub>(4,16)</sub>| | |84.56 |
|IWAE<sub>(8,8)</sub>| | |84.97 |
|IWAE<sub>(16,4)</sub>| | | -|
|IWAE<sub>(64,1)</sub>| |  |86.21|



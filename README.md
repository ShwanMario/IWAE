# Importance Weighted Auto-Encoders Pytorch

Use this code to reproduce the experiments in the Importance Weighted Auto-Encoders(IWAE) paper(2016) by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. The implementation was tested on the MNIST dataset to replicate the result in the above paper. You can train and test VAE and IWAE with 1 or 2 stochastic layers in different configurations of K and M in this repo. 

# Prerequisites for running the code
## MNIST Dataset
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
